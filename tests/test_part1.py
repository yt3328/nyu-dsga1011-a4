import argparse
import copy
import importlib
import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader


PART1_DIR = Path(__file__).resolve().parents[1] / "part1"
sys.path.insert(0, str(PART1_DIR))
main = importlib.import_module("main")
utils = importlib.import_module("utils")
torch.set_num_threads(1)


class TinyTokenizer:
    def __call__(self, texts, padding=None, truncation=None, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [[ord(char) for char in text[:96]] for text in texts]
        return {
            "input_ids": [row + [0] * (96 - len(row)) for row in encoded],
            "attention_mask": [[1] * len(row) + [0] * (96 - len(row)) for row in encoded],
        }

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "synthetic_tokenizer.json").write_text("{}")


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(8, 2)

    def forward(self, input_ids, labels, attention_mask=None):
        hidden = self.dropout(torch.tanh(self.hidden(input_ids.float())))
        logits = self.classifier(hidden)
        return SimpleNamespace(
            logits=logits,
            loss=nn.functional.cross_entropy(logits, labels),
        )

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "synthetic_model.pt")


class InterruptedLoader(DataLoader):
    def __iter__(self):
        self.epochs_started = getattr(self, "epochs_started", 0) + 1
        if self.epochs_started > 1:
            raise RuntimeError("simulated interruption after first completed epoch")
        return super().__iter__()


def training_args(resume_checkpoint=None):
    return argparse.Namespace(
        learning_rate=0.02,
        num_epochs=3,
        batch_size=4,
        resume_checkpoint=resume_checkpoint,
        seed=0,
        debug_train=True,
        train_augmented=False,
    )


def training_loader(loader_class=DataLoader):
    rows = [
        {"input_ids": torch.tensor([float(i % 2), float(i % 3), 1.0]), "labels": i % 2}
        for i in range(12)
    ]
    return loader_class(rows, batch_size=4, shuffle=True)


class TransformationTests(unittest.TestCase):
    def test_deterministic_and_independent_of_labels_and_global_rng(self):
        text = "Excellent performances and beautiful photography make this wonderful movie memorable. " * 10
        first = utils.custom_transform({"text": text, "label": 0})
        random.seed(987)
        for _ in range(100):
            random.random()
        second = utils.custom_transform({"text": text, "label": 1})
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(first["label"], 0)
        self.assertEqual(second["label"], 1)
        self.assertNotEqual(first["text"], text)
        command = (
            "import json,sys; sys.path.insert(0,sys.argv[1]); "
            "from utils import custom_transform; "
            "print(json.dumps(custom_transform({'text':sys.argv[2],'label':0})['text']))"
        )
        result = subprocess.run(
            [sys.executable, "-c", command, str(PART1_DIR), text],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout), first["text"])

    def test_readability_punctuation_negation_and_at_most_one_interior_edit(self):
        text = (
            "I never cannot hardly barely scarcely dislike this movie; don't change NOT or no.\n"
            "Beautiful\tperformances, wonderful cinematography! café 中文 123. "
        ) * 20
        changed = utils.custom_transform({"text": text, "label": 1})["text"]
        self.assertEqual(len(text), len(changed))
        for original, transformed in zip(text, changed):
            if not original.isascii() or not original.isalpha():
                self.assertEqual(original, transformed)
        words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", text)
        changed_words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", changed)
        self.assertEqual(len(words), len(changed_words))
        protected = {"never", "cannot", "hardly", "barely", "scarcely", "don't", "not", "no"}
        for original, transformed in zip(words, changed_words):
            differences = sum(a != b for a, b in zip(original, transformed))
            self.assertLessEqual(differences, 1)
            if len(original) < 4 or original.lower() in protected:
                self.assertEqual(original, transformed)
            if differences:
                self.assertEqual(original[0], transformed[0])
                self.assertEqual(original[-1], transformed[-1])
                self.assertEqual(original.isupper(), transformed.isupper())


class TrainingTests(unittest.TestCase):
    def setUp(self):
        main.device = torch.device("cpu")
        main.tokenizer = TinyTokenizer()
        torch.manual_seed(71)
        random.seed(71)

    def test_updates_parameters_and_has_finite_losses(self):
        model = TinyClassifier()
        initial = copy.deepcopy(model.state_dict())
        losses = []
        handle = model.register_forward_hook(lambda module, inputs, output: losses.append(output.loss.detach().item()))
        with tempfile.TemporaryDirectory() as directory:
            main.do_train(training_args(), model, training_loader(), save_dir=str(Path(directory) / "model"))
            self.assertTrue((Path(directory) / "model" / "last_checkpoint.pt").exists())
        handle.remove()
        self.assertEqual(len(losses), 9)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertTrue(any(not torch.equal(initial[key], model.state_dict()[key]) for key in initial))

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            uninterrupted = TinyClassifier()
            initial = copy.deepcopy(uninterrupted.state_dict())
            random.seed(11)
            torch.manual_seed(11)
            main.do_train(training_args(), uninterrupted, training_loader(), save_dir=str(Path(directory) / "complete"))
            interrupted = TinyClassifier()
            interrupted.load_state_dict(initial)
            random.seed(11)
            torch.manual_seed(11)
            resume_dir = Path(directory) / "resume"
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                main.do_train(training_args(), interrupted, training_loader(InterruptedLoader), save_dir=str(resume_dir))
            checkpoint = resume_dir / "last_checkpoint.pt"
            self.assertTrue(checkpoint.exists())
            resumed = TinyClassifier()
            random.seed(999)
            torch.manual_seed(999)
            main.do_train(training_args(str(checkpoint)), resumed, training_loader(), save_dir=str(resume_dir))
            for key, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)

    def test_evaluation_preserves_alternating_prediction_reference_lines(self):
        model = TinyClassifier()
        rows = [
            {"input_ids": torch.tensor([1.0, 0.0, 1.0]), "labels": 1},
            {"input_ids": torch.tensor([0.0, 1.0, 1.0]), "labels": 0},
            {"input_ids": torch.tensor([1.0, 1.0, 1.0]), "labels": 1},
        ]
        loader = DataLoader(rows, batch_size=2)
        model.eval()
        with torch.no_grad():
            expected_predictions = [int(model(**batch).logits.argmax(-1)[0]) for batch in DataLoader(rows, batch_size=1)]
        expected_lines = [str(value) for prediction, row in zip(expected_predictions, rows) for value in [prediction, row["labels"]]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.txt"
            with mock.patch.object(main.AutoModelForSequenceClassification, "from_pretrained", return_value=model):
                score = main.do_eval(loader, directory, str(path))
            self.assertEqual(path.read_text().splitlines(), expected_lines)
            expected_accuracy = sum(prediction == row["labels"] for prediction, row in zip(expected_predictions, rows)) / len(rows)
            self.assertAlmostEqual(score["accuracy"], expected_accuracy)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        main.tokenizer = TinyTokenizer()

    def test_exactly_5000_unique_training_rows_are_added_with_originals_retained(self):
        size = 5007
        train = Dataset.from_dict({
            "text": [f"{i:06d} Excellent beautiful photography and wonderful performances." for i in range(size)],
            "label": [i % 2 for i in range(size)],
        })
        args = argparse.Namespace(batch_size=8, seed=0, debug_train=False)
        dataset = {"train": train, "test": object(), "unsupervised": object()}
        loader = main.create_augmented_dataloader(args, dataset)
        self.assertEqual(len(loader.dataset), size + 5000)
        original_ids = []
        augmented_ids = []
        for index in range(len(loader.dataset)):
            row = loader.dataset[index]
            self.assertEqual(set(row), {"input_ids", "attention_mask", "labels"})
            identifier = int("".join(chr(int(value)) for value in row["input_ids"][:6]))
            self.assertEqual(int(row["labels"]), identifier % 2)
            (original_ids if index < size else augmented_ids).append(identifier)
        self.assertEqual(original_ids, list(range(size)))
        for index in [0, 17, size - 1]:
            self.assertEqual(loader.dataset[index]["input_ids"].tolist(), main.tokenizer([train[index]["text"]])["input_ids"][0])
        self.assertEqual(len(set(augmented_ids)), 5000)
        self.assertTrue(set(augmented_ids) <= set(original_ids))
        second = main.create_augmented_dataloader(args, dataset)
        for index in [0, size - 1, size, size + 101, size + 4999]:
            self.assertEqual(loader.dataset[index]["input_ids"].tolist(), second.dataset[index]["input_ids"].tolist())
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].shape, (8, 96))
        self.assertEqual(batch["labels"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
