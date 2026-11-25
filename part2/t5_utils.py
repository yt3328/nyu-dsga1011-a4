import os
import json
import shutil
import tempfile
from pathlib import Path

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    import wandb
    wandb.init(project="dsga1011-a4", name=args.experiment_name, config=vars(args))

def initialize_model(args, tokenizer=None):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if not args.finetune:
        raise ValueError("Only the required pretrained T5 fine-tuning baseline is implemented.")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    if tokenizer is not None:
        original_vocab_size = model.get_input_embeddings().num_embeddings
        model.resize_token_embeddings(len(tokenizer))
        args.initial_model_vocab_size = original_vocab_size
        args.tokenizer_vocab_size = len(tokenizer)
        args.model_vocab_size = model.get_input_embeddings().num_embeddings
        print(f"T5 shared vocabulary: {original_vocab_size} -> {args.model_vocab_size}; "
              f"tokenizer entries: {len(tokenizer)}", flush=True)
    if model.config.decoder_start_token_id != 0 or model.config.pad_token_id != 0:
        raise ValueError("Expected T5 decoder-start and padding token IDs to be 0.")
    return model.to(DEVICE)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best, tokenizer=None, settings=None):
    # Save model checkpoint to be able to load the model later
    checkpoint_path = Path(checkpoint_dir) / ("best" if best else "last")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    previous_path = checkpoint_path.with_name(checkpoint_path.name + ".previous")
    if not checkpoint_path.exists() and previous_path.exists():
        previous_path.rename(checkpoint_path)
    stage_path = Path(tempfile.mkdtemp(prefix=checkpoint_path.name + ".stage-", dir=checkpoint_path.parent))
    try:
        model.save_pretrained(stage_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(stage_path)
        if settings is not None:
            with open(stage_path / "settings.json", "w", encoding="utf-8") as settings_file:
                json.dump(settings, settings_file, indent=2)
        if previous_path.exists():
            shutil.rmtree(previous_path)
        if checkpoint_path.exists():
            checkpoint_path.rename(previous_path)
        stage_path.rename(checkpoint_path)
        if previous_path.exists():
            shutil.rmtree(previous_path)
    except BaseException:
        if not checkpoint_path.exists() and previous_path.exists():
            previous_path.rename(checkpoint_path)
        raise
    finally:
        if stage_path.exists():
            shutil.rmtree(stage_path)
    return checkpoint_path

def get_checkpoint_path(args, best):
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir is None:
        model_type = "ft" if args.finetune else "scr"
        checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    checkpoint_path = Path(checkpoint_dir)
    previous_path = checkpoint_path.with_name(checkpoint_path.name + ".previous")
    if not (checkpoint_path / "config.json").is_file() and not (previous_path / "config.json").is_file():
        checkpoint_path = checkpoint_path / ("best" if best else "last")
        previous_path = checkpoint_path.with_name(checkpoint_path.name + ".previous")
    if not (checkpoint_path / "config.json").is_file() and (previous_path / "config.json").is_file():
        checkpoint_path = previous_path
    if not (checkpoint_path / "config.json").is_file():
        raise FileNotFoundError(f"No model configuration found at {checkpoint_path}.")
    return checkpoint_path

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    checkpoint_path = get_checkpoint_path(args, best)
    return T5ForConditionalGeneration.from_pretrained(checkpoint_path, local_files_only=True).to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer_type}")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

