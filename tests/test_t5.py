import argparse
import copy
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import pickle
import random
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, T5TokenizerFast
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


PART2_DIR = Path(__file__).resolve().parents[1] / "part2"
sys.path.insert(0, str(PART2_DIR))
previous_utils = sys.modules.get("utils")
utils_spec = importlib.util.spec_from_file_location("a4_t5_test_sql_utils", PART2_DIR / "utils.py")
sql_utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(sql_utils)
sys.modules["utils"] = sql_utils
load_data = importlib.import_module("load_data")
t5_utils = importlib.import_module("t5_utils")
train_t5 = importlib.import_module("train_t5")
if previous_utils is not None:
    sys.modules["utils"] = previous_utils
torch.set_num_threads(1)


class SyntheticTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 512
    name_or_path = "synthetic-no-download"

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, **kwargs):
        if isinstance(text, list):
            encoded = [self(value, add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length)["input_ids"] for value in text]
            return {"input_ids": encoded, "attention_mask": [[1] * len(row) for row in encoded]}
        encoded = [ord(character) + 2 for character in text]
        if add_special_tokens:
            encoded.append(self.eos_token_id)
        if truncation and max_length is not None:
            encoded = encoded[:max_length]
        return {"input_ids": encoded, "attention_mask": [1] * len(encoded)}

    def encode(self, text, **kwargs):
        return self(text, **kwargs)["input_ids"]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return "".join(chr(int(value) - 2) for value in ids if int(value) > 1)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "synthetic_tokenizer.json").write_text("{}")


class SyntheticDataset(Dataset):
    def __init__(self, include_targets=True):
        self.tokenizer = SyntheticTokenizer()
        self.rows = [
            {"encoder_ids": torch.tensor([2, 3, 1]), "decoder_start_id": 0, "decoder_targets": torch.tensor([7, 8, 1])},
            {"encoder_ids": torch.tensor([4, 1]), "decoder_start_id": 0, "decoder_targets": torch.tensor([9, 1])},
            {"encoder_ids": torch.tensor([5, 6, 1]), "decoder_start_id": 0, "decoder_targets": torch.tensor([10, 11, 1])},
        ]
        if not include_targets:
            self.rows = [{key: value for key, value in row.items() if key != "decoder_targets"} for row in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def tiny_model(dropout_rate=0.0):
    return T5ForConditionalGeneration(T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        d_kv=8,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=dropout_rate,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    ))


def synthetic_args(**overrides):
    values = {
        "finetune": True,
        "use_wandb": False,
        "max_new_tokens": 6,
        "num_beams": 1,
        "max_target_length": 32,
        "max_source_length": 32,
        "experiment_name": "synthetic",
        "checkpoint_dir": None,
        "output_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class InterruptedLoader(DataLoader):
    def __iter__(self):
        self.epochs_started = getattr(self, "epochs_started", 0) + 1
        if self.epochs_started > 1:
            raise RuntimeError("simulated interruption after first completed epoch")
        return super().__iter__()


class DataTests(unittest.TestCase):
    def test_sql_operators_survive_decoding_execution_and_saved_tokenizer_reload(self):
        queries = [f"SELECT x FROM sample WHERE x {operator} 2" for operator in ("<", "<=", "<>")]
        # A real fast-tokenizer backend with the same missing '<' failure, without
        # fetching or depending on a cached pretrained tokenizer in offline tests.
        vocabulary = {"<pad>": 0, "</s>": 1, "<unk>": 2}
        for character in "".join(queries):
            if character != "<" and character not in vocabulary:
                vocabulary[character] = len(vocabulary)
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        backend.decoder = decoders.Fuse()
        tokenizer = T5TokenizerFast(tokenizer_object=backend, extra_ids=0)
        native_ids = tokenizer(queries)["input_ids"]
        self.assertTrue(all(tokenizer.unk_token_id in row for row in native_ids))
        with mock.patch.object(load_data.T5TokenizerFast, "from_pretrained", return_value=tokenizer):
            extended = load_data.create_t5_tokenizer()
        ids = extended(queries)["input_ids"]
        self.assertTrue(all(extended.unk_token_id not in row for row in ids))
        decoded = extended.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(decoded, queries)
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE sample (x INTEGER)")
            connection.executemany("INSERT INTO sample VALUES (?)", [(1,), (2,), (3,)])
            self.assertEqual([connection.execute(query).fetchall() for query in decoded],
                             [[(1,)], [(1,), (2,)], [(1,), (3,)]])
        with tempfile.TemporaryDirectory() as directory:
            extended.save_pretrained(directory)
            restored = AutoTokenizer.from_pretrained(directory, local_files_only=True)
            self.assertEqual(restored(queries)["input_ids"], ids)
            self.assertEqual(restored.batch_decode(ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False), queries)

    def test_dynamic_padding_shift_start_and_test_shapes(self):
        dataset = SyntheticDataset()
        encoder_ids, encoder_mask, decoder_inputs, decoder_targets, starts = load_data.normal_collate_fn([dataset[0], dataset[1]])
        self.assertEqual(encoder_ids.tolist(), [[2, 3, 1], [4, 1, 0]])
        self.assertEqual(encoder_mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(decoder_inputs.tolist(), [[0, 7, 8], [0, 9, 1]])
        self.assertEqual(decoder_targets.tolist(), [[7, 8, 1], [9, 1, 0]])
        self.assertEqual(starts.tolist(), [[0], [0]])
        test_dataset = SyntheticDataset(include_targets=False)
        test_batch = load_data.test_collate_fn([test_dataset[0], test_dataset[1]])
        self.assertEqual(len(test_batch), 3)
        self.assertEqual(test_batch[0].tolist(), encoder_ids.tolist())
        self.assertEqual(test_batch[1].tolist(), encoder_mask.tolist())
        self.assertEqual(test_batch[2].tolist(), starts.tolist())

    def test_dataset_preserves_alignment_literals_eos_and_test_without_sql(self):
        tokenizer = SyntheticTokenizer()
        inputs = ["find flights from BOS", "show flights"]
        targets = ["SELECT origin FROM flight WHERE code = 'BOS';", "SELECT * FROM flight WHERE x >= 2;"]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "train.nl").write_text("\n".join(inputs) + "\n")
            (folder / "train.sql").write_text("\n".join(targets) + "\n")
            (folder / "test.nl").write_text("\n".join(inputs) + "\n")
            dataset = load_data.T5Dataset(directory, "train", tokenizer=tokenizer)
            self.assertEqual(len(dataset), 2)
            for index in range(2):
                self.assertEqual(tokenizer.decode(dataset[index]["encoder_ids"]), inputs[index])
                self.assertEqual(tokenizer.decode(dataset[index]["decoder_targets"]), targets[index])
                self.assertEqual(int(dataset[index]["decoder_targets"][-1]), 1)
                self.assertEqual(dataset[index]["decoder_start_id"], 0)
            test_dataset = load_data.T5Dataset(directory, "test", tokenizer=tokenizer)
            self.assertEqual(len(test_dataset), 2)
            self.assertNotIn("decoder_targets", test_dataset[0])
            (folder / "train.sql").write_text(targets[0] + "\n")
            with self.assertRaises(ValueError):
                load_data.T5Dataset(directory, "train", tokenizer=tokenizer)


class TinyT5Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        train_t5.DEVICE = torch.device("cpu")
        t5_utils.DEVICE = torch.device("cpu")

    def test_pretrained_initialization_resizes_tied_embeddings_before_training(self):
        model = tiny_model()
        tokenizer = mock.MagicMock()
        tokenizer.__len__.return_value = 35
        args = synthetic_args()
        with mock.patch.object(t5_utils.T5ForConditionalGeneration, "from_pretrained", return_value=model):
            resized = t5_utils.initialize_model(args, tokenizer=tokenizer)
        self.assertEqual(resized.config.vocab_size, 35)
        self.assertEqual(args.initial_model_vocab_size, 32)
        self.assertEqual(args.model_vocab_size, 35)
        self.assertEqual(resized.get_input_embeddings().weight.data_ptr(), resized.lm_head.weight.data_ptr())
        loss = resized(input_ids=torch.tensor([[34, 1]]), labels=torch.tensor([[33, 1]])).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        with tempfile.TemporaryDirectory() as directory:
            t5_utils.save_model(directory, resized, best=True)
            restored = t5_utils.load_model_from_checkpoint(synthetic_args(checkpoint_dir=directory), best=True)
            self.assertEqual(restored.config.vocab_size, 35)
            self.assertEqual(restored.get_input_embeddings().weight.data_ptr(), restored.lm_head.weight.data_ptr())
            for key, value in resized.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_finite_backward_updates_and_tiny_training_loss_decreases(self):
        model = tiny_model()
        initial = copy.deepcopy(model.state_dict())
        loader = DataLoader(SyntheticDataset(), batch_size=3, collate_fn=load_data.normal_collate_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)
        losses = []
        with mock.patch.object(train_t5, "tqdm", side_effect=lambda iterable, **kwargs: iterable):
            for _ in range(30):
                losses.append(train_t5.train_epoch(synthetic_args(), model, loader, optimizer, None))
        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertLess(losses[-1], losses[0] * 0.8)
        self.assertTrue(any(not torch.equal(initial[key], model.state_dict()[key]) for key in initial))

    def test_padded_targets_do_not_contribute_to_loss(self):
        model = tiny_model()
        loader = DataLoader(SyntheticDataset(), batch_size=3, collate_fn=load_data.normal_collate_fn)
        encoder, mask, decoder_inputs, targets, _ = next(iter(loader))
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=encoder, attention_mask=mask, decoder_input_ids=decoder_inputs).logits
            expected = torch.nn.functional.cross_entropy(logits[targets != 0], targets[targets != 0]).item()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        with mock.patch.object(train_t5, "tqdm", side_effect=lambda iterable, **kwargs: iterable):
            actual = train_t5.train_epoch(synthetic_args(), model, loader, optimizer, None)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_checkpoint_reloads_independent_logits_and_generation(self):
        model = tiny_model()
        model.eval()
        encoder = torch.tensor([[2, 3, 1]])
        decoder = torch.tensor([[0, 7, 8]])
        with torch.no_grad():
            expected_logits = model(input_ids=encoder, decoder_input_ids=decoder).logits
            expected_generated = model.generate(input_ids=encoder, max_new_tokens=5)
        with tempfile.TemporaryDirectory() as directory:
            t5_utils.save_model(directory, model, best=True)
            t5_utils.save_model(directory, model, best=False)
            args = synthetic_args(checkpoint_dir=directory)
            restored = t5_utils.load_model_from_checkpoint(args, best=True)
            restored.eval()
            with torch.no_grad():
                actual_logits = restored(input_ids=encoder, decoder_input_ids=decoder).logits
                actual_generated = restored.generate(input_ids=encoder, max_new_tokens=5)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
            self.assertTrue(torch.equal(actual_generated, expected_generated))
            last = t5_utils.load_model_from_checkpoint(args, best=False)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(last.state_dict()[key], value, rtol=0, atol=0)

    def test_checkpoint_survives_failed_save_and_loads_previous_directory(self):
        model = tiny_model()
        expected_state = copy.deepcopy(model.state_dict())
        tokenizer = SyntheticTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            t5_utils.save_model(directory, model, best=True, tokenizer=tokenizer, settings={"selection": "old"})
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            with mock.patch.object(tokenizer, "save_pretrained", side_effect=OSError("simulated tokenizer save failure")):
                with self.assertRaisesRegex(OSError, "simulated tokenizer save failure"):
                    t5_utils.save_model(directory, model, best=True, tokenizer=tokenizer, settings={"selection": "new"})
            args = synthetic_args(checkpoint_dir=directory)
            restored = t5_utils.load_model_from_checkpoint(args, best=True)
            for key, value in expected_state.items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(json.loads((Path(directory) / "best" / "settings.json").read_text()), {"selection": "old"})
            self.assertFalse(list(Path(directory).glob("best.stage-*")))
            (Path(directory) / "best").rename(Path(directory) / "best.previous")
            restored = t5_utils.load_model_from_checkpoint(args, best=True)
            for key, value in expected_state.items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_training_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data_dir = folder / "data"
            data_dir.mkdir()
            for filename in ["train.nl", "train.sql", "dev.nl", "dev.sql"]:
                (data_dir / filename).write_text("synthetic aligned input\n" * 3)

            def args_for_run(name, resume_checkpoint=None):
                run_dir = folder / name
                return synthetic_args(
                    run_dir=str(run_dir), checkpoint_dir=str(run_dir / "checkpoints"),
                    data_dir=str(data_dir), ground_truth_records=str(folder / "unused_ground_truth.pkl"),
                    learning_rate=0.01, weight_decay=0.0, scheduler_type="linear", num_warmup_epochs=0,
                    max_n_epochs=3, patience_epochs=5, batch_size=2, optimizer_type="AdamW", seed=31,
                    resume_checkpoint=resume_checkpoint,
                )

            def reset_seed(value):
                random.seed(value)
                np.random.seed(value)
                torch.manual_seed(value)

            def loader(loader_class=DataLoader):
                return loader_class(SyntheticDataset(), batch_size=2, shuffle=True, collate_fn=load_data.normal_collate_fn)

            uninterrupted = tiny_model(dropout_rate=0.1)
            initial_state = copy.deepcopy(uninterrupted.state_dict())
            complete_args = args_for_run("complete")
            complete_loader = loader()
            optimizer, scheduler = t5_utils.initialize_optimizer_and_scheduler(complete_args, uninterrupted, len(complete_loader))
            with mock.patch.object(train_t5, "eval_epoch", return_value=(1.0, 0.5, 0.4, 0.3, 0.0)), mock.patch.object(train_t5, "tqdm", side_effect=lambda iterable, **kwargs: iterable), contextlib.redirect_stdout(io.StringIO()):
                reset_seed(31)
                train_t5.train(complete_args, uninterrupted, complete_loader, complete_loader, optimizer, scheduler)
                interrupted = tiny_model(dropout_rate=0.1)
                interrupted.load_state_dict(initial_state)
                interrupted_args = args_for_run("resume")
                interrupted_loader = loader(InterruptedLoader)
                optimizer, scheduler = t5_utils.initialize_optimizer_and_scheduler(interrupted_args, interrupted, len(interrupted_loader))
                reset_seed(31)
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_t5.train(interrupted_args, interrupted, interrupted_loader, complete_loader, optimizer, scheduler)
                resume_checkpoint = Path(interrupted_args.checkpoint_dir) / "training_state.pt"
                first_epoch = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(first_epoch["epoch"], 1)
                resumed = tiny_model(dropout_rate=0.1)
                resumed_args = args_for_run("resume", str(resume_checkpoint))
                resumed_loader = loader()
                optimizer, scheduler = t5_utils.initialize_optimizer_and_scheduler(resumed_args, resumed, len(resumed_loader))
                reset_seed(999)
                train_t5.train(resumed_args, resumed, resumed_loader, complete_loader, optimizer, scheduler)
            for key, value in uninterrupted.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
            complete_state = torch.load(Path(complete_args.checkpoint_dir) / "training_state.pt", map_location="cpu", weights_only=False)
            resumed_state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(resumed_state["epoch"], 3)
            self.assertEqual(resumed_state["scheduler_state_dict"], complete_state["scheduler_state_dict"])
            self.assertEqual([row["train_loss"] for row in resumed_state["history"]], [row["train_loss"] for row in complete_state["history"]])

    def test_eval_metric_order_source_only_generation_and_export_counts(self):
        model = tiny_model()
        dataset = SyntheticDataset()
        loader = DataLoader(dataset, batch_size=2, collate_fn=load_data.normal_collate_fn)
        original_generate = model.generate
        calls = []

        def generation_spy(*args, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            self.assertNotIn("labels", kwargs)
            if "decoder_input_ids" in kwargs:
                self.assertEqual(kwargs["decoder_input_ids"].shape[-1], 1)
                self.assertTrue(torch.all(kwargs["decoder_input_ids"] == 0))
            calls.append(kwargs["input_ids"].shape[0])
            return original_generate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            database = folder / "test.db"
            sqlite3.connect(database).close()
            output_sql = folder / "generated.sql"
            output_records = folder / "generated.pkl"
            gt_sql = folder / "reference.sql"
            gt_sql.write_text("SELECT 1;\n" * 3)
            gt_records = folder / "reference.pkl"
            with gt_records.open("wb") as handle:
                pickle.dump(([[(1,)]] * 3, [""] * 3), handle)
            with mock.patch.object(model, "generate", side_effect=generation_spy), mock.patch.object(train_t5, "compute_metrics", return_value=(0.1, 0.2, 0.3, ["", "synthetic SQL error", ""])), mock.patch.object(sql_utils, "DB_PATH", str(database)):
                result = train_t5.eval_epoch(synthetic_args(), model, loader, str(gt_sql), str(output_sql), str(gt_records), str(output_records))
            self.assertTrue(math.isfinite(result[0]))
            self.assertEqual(result[1:4], (0.3, 0.2, 0.1))
            self.assertAlmostEqual(result[4], 1 / 3)
            self.assertEqual(calls, [2, 1])
            self.assertEqual(len(output_sql.read_text().splitlines()), 3)
            with output_records.open("rb") as handle:
                records, errors = pickle.load(handle)
            self.assertEqual(len(records), 3)
            self.assertEqual(len(errors), 3)

    def test_unlabeled_test_export_counts_overwrite_and_missing_output_guards(self):
        model = tiny_model()
        dataset = SyntheticDataset(include_targets=False)
        loader = DataLoader(dataset, batch_size=2, collate_fn=load_data.test_collate_fn)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            database = folder / "test.db"
            sqlite3.connect(database).close()
            output_sql = folder / "test.sql"
            output_records = folder / "test.pkl"
            args = synthetic_args(checkpoint_dir=str(folder / "checkpoint"))
            t5_utils.save_model(args.checkpoint_dir, model, best=True)
            with mock.patch.object(sql_utils, "DB_PATH", str(database)):
                train_t5.test_inference(args, model, loader, str(output_sql), str(output_records))
            self.assertEqual(len(output_sql.read_text().splitlines()), len(dataset))
            with output_records.open("rb") as handle:
                records, errors = pickle.load(handle)
            self.assertEqual(len(records), len(dataset))
            self.assertEqual(len(errors), len(dataset))
            metadata = json.loads(Path(str(output_sql) + ".generation.json").read_text())
            self.assertEqual(metadata["count"], len(dataset))
            selected_checkpoint = Path(args.checkpoint_dir) / "best"
            self.assertEqual(metadata["checkpoint_dir"], str(selected_checkpoint.resolve()))
            self.assertEqual(metadata["checkpoint_sha256"]["config.json"], hashlib.sha256((selected_checkpoint / "config.json").read_bytes()).hexdigest())
            original_bytes = output_sql.read_bytes()
            with self.assertRaises(FileExistsError):
                train_t5.test_inference(args, model, loader, str(output_sql), str(output_records))
            self.assertEqual(output_sql.read_bytes(), original_bytes)
            with mock.patch.object(dataset.tokenizer, "batch_decode", return_value=[]):
                with self.assertRaises(ValueError):
                    train_t5.test_inference(args, model, loader, str(folder / "missing.sql"), str(folder / "missing.pkl"))
            self.assertFalse((folder / "missing.sql").exists())


if __name__ == "__main__":
    unittest.main()
