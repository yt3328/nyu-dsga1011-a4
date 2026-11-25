import os
import argparse
import json
import random
import hashlib
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb, get_checkpoint_path
from transformers import GenerationConfig
from load_data import load_t5_data, get_dataloader
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', default=True, help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--mode', choices=['train', 'dev', 'test', 'smoke'], default='train')
    parser.add_argument('--run_dir', default='runs/t5_baseline')
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--resume_checkpoint', default=None)
    parser.add_argument('--decoding_config', default=None)
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent / 'data'))
    parser.add_argument('--ground_truth_records', default=str(Path(__file__).resolve().parent / 'records/ground_truth_dev.pkl'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_source_length', type=int, default=None)
    parser.add_argument('--max_target_length', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--smoke_steps', type=int, default=30)
    parser.add_argument('--test_sql_path', default=str(Path(__file__).resolve().parent / 'results/t5_ft_experiment_test.sql'))
    parser.add_argument('--test_record_path', default=str(Path(__file__).resolve().parent / 'records/t5_ft_experiment_test.pkl'))

    args = parser.parse_args()
    if min(args.max_n_epochs, args.patience_epochs, args.batch_size, args.test_batch_size,
           args.max_new_tokens, args.num_beams, args.smoke_steps) < 1 or args.learning_rate <= 0:
        parser.error('Epochs, patience, batch sizes, generation limits, and learning rate must be positive.')
    if not 0 <= args.num_warmup_epochs < args.max_n_epochs:
        parser.error('Warmup epochs must be nonnegative and smaller than max_n_epochs.')
    if args.resume_checkpoint and args.mode != 'train':
        parser.error('--resume_checkpoint requires --mode train.')
    args.checkpoint_dir = args.checkpoint_dir or str(Path(args.run_dir) / 'checkpoints')
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = args.checkpoint_dir
    gt_sql_path = str(Path(args.data_dir) / 'dev.sql')
    gt_record_path = args.ground_truth_records
    run_dir = Path(args.run_dir)
    start_epoch = 0
    history = []
    config_keys = ['learning_rate', 'weight_decay', 'scheduler_type', 'num_warmup_epochs',
                   'max_n_epochs', 'patience_epochs', 'batch_size', 'seed',
                   'max_source_length', 'max_target_length', 'max_new_tokens', 'num_beams']
    training_config = {key: getattr(args, key) for key in config_keys}
    training_config['data_sha256'] = {
        name: hashlib.sha256((Path(args.data_dir) / name).read_bytes()).hexdigest()
        for name in ['train.nl', 'train.sql', 'dev.nl', 'dev.sql']
    }
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        if state['training_config'] != training_config:
            raise ValueError('Resume settings differ from the saved training run.')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch']
        best_f1 = state['best_f1']
        epochs_since_improvement = state['epochs_since_improvement']
        history = state['history']
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        torch.set_rng_state(state['torch_rng_state'])
        if torch.cuda.is_available() and state['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    for epoch in range(start_epoch, args.max_n_epochs):
        if epochs_since_improvement >= args.patience_epochs:
            break
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        epoch_dir = run_dir / f'epoch_{epoch + 1:03d}'
        attempt = 1
        while epoch_dir.exists():
            epoch_dir = run_dir / f'epoch_{epoch + 1:03d}_retry_{attempt:03d}'
            attempt += 1
        epoch_dir.mkdir(parents=True)
        model_sql_path = str(epoch_dir / 'dev.sql')
        model_record_path = str(epoch_dir / 'dev.pkl')
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            import wandb
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        epoch_result = {'epoch': epoch + 1, 'train_loss': tr_loss, 'dev_loss': eval_loss,
                        'record_f1': record_f1, 'record_em': record_em,
                        'sql_em': sql_em, 'error_rate': error_rate,
                        'predictions_dir': str(epoch_dir)}
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        tokenizer = train_loader.dataset.tokenizer
        save_model(checkpoint_dir, model, best=False, tokenizer=tokenizer, settings=vars(args))
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True, tokenizer=tokenizer, settings=vars(args))
        state = {
            'epoch': epoch + 1, 'best_f1': best_f1,
            'epochs_since_improvement': epochs_since_improvement, 'history': history,
            'training_config': training_config, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary_state = Path(checkpoint_dir) / 'training_state.pt.tmp'
        torch.save(state, temporary_state)
        os.replace(temporary_state, Path(checkpoint_dir) / 'training_state.pt')
        with open(run_dir / 'history.json.tmp', 'w') as history_file:
            json.dump(history, history_file, indent=2)
        os.replace(run_dir / 'history.json.tmp', run_dir / 'history.json')

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite T5 training loss.')
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        raise ValueError('Training requires nonpadding target tokens.')
    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    # TODO
    model.eval()
    tokenizer = dev_loader.dataset.tokenizer
    total_loss = 0.0
    total_tokens = 0
    generated_queries = []
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_input in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            logits = model(input_ids=encoder_input, attention_mask=encoder_mask,
                           decoder_input_ids=decoder_input).logits
            non_pad = decoder_targets != PAD_IDX
            token_count = non_pad.sum().item()
            loss_sum = nn.functional.cross_entropy(logits[non_pad], decoder_targets[non_pad], reduction='sum')
            total_loss += loss_sum.item()
            total_tokens += token_count
            generated = model.generate(input_ids=encoder_input, attention_mask=encoder_mask,
                                       decoder_input_ids=initial_input.to(DEVICE),
                                       max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                                       do_sample=False)
            generated_queries.extend(tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False))
    if total_tokens == 0 or len(generated_queries) != len(dev_loader.dataset):
        raise ValueError('Development output count or token count is invalid.')
    generated_queries = [query.replace('\r', ' ').replace('\n', ' ') for query in generated_queries]
    Path(model_sql_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_record_path).parent.mkdir(parents=True, exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, errors = compute_metrics(gt_sql_pth, model_sql_path,
                                                       gt_record_path, model_record_path)
    if len(errors) != len(generated_queries):
        raise ValueError('Development execution error count differs from output count.')
    error_rate = sum(bool(error) for error in errors) / len(errors)
    return total_loss / total_tokens, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    if Path(model_sql_path).exists() or Path(model_record_path).exists():
        raise FileExistsError('Test export already exists. Preserve it and choose new output paths.')
    generated_queries = []
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_input in tqdm(test_loader):
            generated = model.generate(input_ids=encoder_input.to(DEVICE),
                                       attention_mask=encoder_mask.to(DEVICE),
                                       decoder_input_ids=initial_input.to(DEVICE),
                                       max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                                       do_sample=False)
            generated_queries.extend(test_loader.dataset.tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    if len(generated_queries) != len(test_loader.dataset):
        raise ValueError('Test query count differs from the input count.')
    generated_queries = [query.replace('\r', ' ').replace('\n', ' ') for query in generated_queries]
    Path(model_sql_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_record_path).parent.mkdir(parents=True, exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    checkpoint_path = get_checkpoint_path(args, best=True) if args.checkpoint_dir else None
    file_hashes = {}
    if checkpoint_path is not None:
        for path in sorted(checkpoint_path.rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with open(path, 'rb') as checkpoint_file:
                    for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
                        digest.update(chunk)
                file_hashes[str(path.relative_to(checkpoint_path))] = digest.hexdigest()
    with open(str(model_sql_path) + '.generation.json', 'x') as generation_file:
        json.dump({'checkpoint_dir': str(checkpoint_path.resolve()) if checkpoint_path else None,
                   'checkpoint_sha256': file_hashes,
                   'max_new_tokens': args.max_new_tokens, 'num_beams': args.num_beams,
                   'do_sample': False, 'count': len(generated_queries)}, generation_file, indent=2)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_dir = Path(args.run_dir)
    if args.mode in ('train', 'smoke'):
        if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_checkpoint:
            raise FileExistsError(f'Run already exists: {run_dir}. Use a fresh directory or resume.')
        checkpoint_dir = Path(args.checkpoint_dir)
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not args.resume_checkpoint:
            raise FileExistsError(f'Checkpoint directory already exists: {checkpoint_dir}.')
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'args.json', 'w') as args_file:
            json.dump(vars(args), args_file, indent=2)

    # Load the data and the model
    if args.mode in ('train', 'smoke'):
        train_loader, dev_loader, _ = load_t5_data(
            args.batch_size, args.test_batch_size, data_folder=args.data_dir,
            max_source_length=args.max_source_length, max_target_length=args.max_target_length)
        longest_target = max(len(row['decoder_targets']) for loader in [train_loader, dev_loader]
                             for row in loader.dataset)
        if args.max_new_tokens < longest_target:
            raise ValueError(f'max_new_tokens={args.max_new_tokens} is below the longest target ({longest_target}).')
        unknown_id = getattr(train_loader.dataset.tokenizer, 'unk_token_id', None)
        if unknown_id is not None and any(
                unknown_id in row['decoder_targets'] for loader in [train_loader, dev_loader]
                for row in loader.dataset):
            raise ValueError('SQL targets contain unknown tokenizer tokens; inspect token_stats.json before training.')
        model = initialize_model(args, tokenizer=train_loader.dataset.tokenizer)
        if model.config.decoder_start_token_id != 0:
            raise ValueError('This baseline expects T5 decoder_start_token_id=0.')
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

        # Train
        if args.mode == 'smoke':
            batch = next(iter(train_loader))
            losses = []
            for _ in range(args.smoke_steps):
                losses.append(train_epoch(args, model, [batch], optimizer, None))
            save_model(args.checkpoint_dir, model, best=True,
                       tokenizer=train_loader.dataset.tokenizer, settings=vars(args))
            restored = load_model_from_checkpoint(args, best=True)
            restored.eval()
            model.eval()
            with torch.no_grad():
                inputs = {'input_ids': batch[0].to(DEVICE), 'attention_mask': batch[1].to(DEVICE),
                          'decoder_input_ids': batch[2].to(DEVICE)}
                torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits)
                generated = restored.generate(input_ids=batch[0].to(DEVICE), attention_mask=batch[1].to(DEVICE),
                                             max_new_tokens=args.max_new_tokens, do_sample=False)
            if not np.isfinite(losses).all():
                raise FloatingPointError('Nonfinite smoke-test losses.')
            result = {'synthetic': False, 'tiny_training_batch': True, 'steps': args.smoke_steps,
                      'first_loss': losses[0], 'last_loss': losses[-1],
                      'loss_decreased': losses[-1] < losses[0],
                      'generated_count': len(generated), 'checkpoint_reload_equal': True}
            with open(run_dir / 'smoke.json', 'x') as smoke_file:
                json.dump(result, smoke_file, indent=2)
            print(json.dumps(result), flush=True)
            return
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        return

    # Evaluate
    from transformers import AutoTokenizer
    model = load_model_from_checkpoint(args, best=True)
    checkpoint_path = get_checkpoint_path(args, best=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), local_files_only=True)
    decoding_keys = ['max_new_tokens', 'num_beams', 'max_source_length', 'max_target_length']
    if args.mode == 'test':
        settings_path = Path(args.decoding_config) if args.decoding_config else checkpoint_path / 'settings.json'
        with open(settings_path) as settings_file:
            frozen_settings = json.load(settings_file)
        for key in decoding_keys:
            if getattr(args, key) != frozen_settings[key]:
                raise ValueError(f'Final test {key} differs from the frozen settings at {settings_path}.')
    model.eval()
    if args.mode == 'dev':
        # Dev set
        loader = get_dataloader(args.test_batch_size, 'dev', data_folder=args.data_dir, tokenizer=tokenizer,
                                max_source_length=args.max_source_length, max_target_length=args.max_target_length)
        output_dir = run_dir / 'selected_dev'
        output_dir.mkdir(parents=True, exist_ok=False)
        values = eval_epoch(args, model, loader, str(Path(args.data_dir) / 'dev.sql'),
                            str(output_dir / 'dev.sql'), args.ground_truth_records, str(output_dir / 'dev.pkl'))
        result = dict(zip(['loss', 'record_f1', 'record_em', 'sql_em', 'error_rate'], values))
        with open(output_dir / 'metrics.json', 'x') as metrics_file:
            json.dump(result, metrics_file, indent=2)
        with open(output_dir / 'decoding_config.json', 'x') as decoding_file:
            json.dump({key: getattr(args, key) for key in decoding_keys}, decoding_file, indent=2)
        print(json.dumps(result), flush=True)
    else:
        # Test set
        loader = get_dataloader(args.test_batch_size, 'test', data_folder=args.data_dir, tokenizer=tokenizer,
                                max_source_length=args.max_source_length)
        test_inference(args, model, loader, args.test_sql_path, args.test_record_path)

if __name__ == "__main__":
    main()
