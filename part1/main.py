import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import random
import argparse
from utils import *
import os
import json
from pathlib import Path
import numpy as np

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,
                     max_length=getattr(tokenizer, "model_max_length", 512))


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    save_path = Path(save_dir)
    resume_path = getattr(args, "resume_checkpoint", None)
    if save_path.exists() and any(save_path.iterdir()) and not resume_path:
        raise FileExistsError(f"Training output already exists: {save_path}. Use a fresh directory or resume.")
    if not len(train_dataloader):
        raise ValueError("Training requires at least one batch.")
    save_path.mkdir(parents=True, exist_ok=True)
    model_device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Implement the training loop --- make sure to use the optimizer and lr_sceduler (learning rate scheduler)
    # Remember that pytorch uses gradient accumumlation so you need to use zero_grad (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
    # You can use progress_bar.update(1) to see the progress during training
    # You can refer to the pytorch tutorial covered in class for reference

    run_config = {
        "model": "bert-base-cased",
        "learning_rate": args.learning_rate,
        "num_epochs": num_epochs,
        "batch_size": args.batch_size,
        "seed": getattr(args, "seed", 0),
        "max_length": getattr(args, "max_length", 512),
        "debug_train": getattr(args, "debug_train", False),
        "train_augmented": getattr(args, "train_augmented", False),
        "training_examples": len(train_dataloader.dataset),
        "batches_per_epoch": len(train_dataloader),
        "transformation": transformation_settings(),
    }
    start_epoch = 0
    history = []
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["run_config"] != run_config:
            raise ValueError("Resume configuration differs from the saved run.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        progress_bar.update(start_epoch * len(train_dataloader))

    with open(save_path / "run_config.json", "w") as config_file:
        json.dump(run_config, config_file, indent=2)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        example_count = 0
        for batch in train_dataloader:
            batch = {key: value.to(model_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in epoch {epoch + 1}.")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            batch_size = batch["labels"].shape[0]
            total_loss += loss.item() * batch_size
            example_count += batch_size
            progress_bar.update(1)

        epoch_result = {"epoch": epoch + 1, "train_loss": total_loss / example_count}
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "run_config": run_config,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary_checkpoint = save_path / "last_checkpoint.pt.tmp"
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, save_path / "last_checkpoint.pt")
        with open(save_path / "train_history.json.tmp", "w") as history_file:
            json.dump(history, history_file, indent=2)
        os.replace(save_path / "train_history.json.tmp", save_path / "train_history.json")
    progress_bar.close()

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)
    if "tokenizer" in globals() and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)

    return history


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    output_path = Path(out_file)
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    correct = 0
    count = 0

    with open(temporary_output, "w") as output_handle:
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            count += batch["labels"].numel()

            # write to output file
            for pred, label in zip(predictions, batch["labels"]):
                output_handle.write(f"{pred.item()}\n")
                output_handle.write(f"{label.item()}\n")
    if count == 0:
        raise ValueError("Evaluation requires at least one example.")
    os.replace(temporary_output, output_path)
    score = {"accuracy": correct / count}
    with open(str(output_path) + ".metrics.json", "x") as metrics_file:
        json.dump({"accuracy": score["accuracy"], "correct": correct, "count": count,
                   "model_dir": str(output_dir), "transformation": transformation_settings()
                   if output_path.stem.endswith("transformed") else None}, metrics_file, indent=2)

    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(args, dataset):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Here, 'dataset' is the original dataset. You should return a dataloader called 'train_dataloader' -- this
    # dataloader will be for the original training split augmented with 5k random transformed examples from the training set.
    # You may find it helpful to see how the dataloader was created at other place in this code.

    original_train = dataset["train"]
    if len(original_train) < 5000:
        raise ValueError("Augmentation requires at least 5,000 labeled training examples.")
    augmented_examples = original_train.shuffle(seed=42).select(range(5000))
    augmented_examples = augmented_examples.map(custom_transform, load_from_cache_file=False)
    augmented_train = datasets.concatenate_datasets([original_train, augmented_examples])
    assert len(augmented_train) == len(original_train) + 5000
    tokenized_train = augmented_train.map(tokenize_function, batched=True, load_from_cache_file=False)
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_train.set_format("torch")
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=args.batch_size)

    ##### YOUR CODE ENDS HERE ######

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        return None

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    if args.train and args.train_augmented:
        parser.error("Run original and augmented training in separate processes.")
    if args.debug_train and args.train_augmented:
        parser.error("The augmented experiment must retain the full original training split.")
    if args.resume_checkpoint and not (args.train or args.train_augmented):
        parser.error("--resume_checkpoint requires a training mode.")
    if args.num_epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("num_epochs, batch_size, and learning_rate must be positive.")
    if not 1 <= args.max_length <= 512:
        parser.error("max_length must be between 1 and BERT's 512-token limit.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.debug_transformation:
        dataset = load_dataset("imdb", split="train").train_test_split(test_size=5, seed=42)
        create_transformed_dataloader(args, dataset, True)
        raise SystemExit(0)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenizer.model_max_length = args.max_length

    # Tokenize the dataset
    dataset = load_dataset("imdb")
    dataset = datasets.DatasetDict({"train": dataset["train"], "test": dataset["test"]})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir=args.output_dir or "./out")
        # Change eval dir
        args.model_dir = args.output_dir or "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir=args.output_dir or "./out_augmented")
        # Change eval dir
        args.model_dir = args.output_dir or "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)
