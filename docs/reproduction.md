# Reproduction instructions

BERT sentiment classification and T5 text-to-SQL. This implementation is based on the [course starter repository](https://github.com/gregdurrett/nyu-dsga1011-a4). Original instructions are retained in [STARTER_README.md](../STARTER_README.md) and the two part directories.

## Setup

`part1/` contains BERT training, evaluation and the fixed keyboard-typo transformation. `part2/` contains T5 training, generation, database evaluation and error analysis. The supplied flight inputs, SQL references, database/schema and development records are included unchanged. Hugging Face downloads IMDB and the initial public models on first use. Model weights and generated predictions are distributed separately.

Use separate Python 3.11 environments: the two parts pin incompatible Transformers versions. Dependency files are unchanged. The full Part 2 environment was used on Linux; its pinned bitsandbytes package is not distributed for macOS. Full fine-tuning does not use quantization. The keyboard transformation requires no NLTK corpus downloads.

```bash
python3.11 -m venv .venv-part1
.venv-part1/bin/python -m pip install -r part1/requirements.txt
python3.11 -m venv .venv-part2
.venv-part2/bin/python -m pip install -r part2/requirements.txt
```

Run these reproduction commands from the repository root, using fresh output directories. The submitted experiments and predictions are already complete. Each training or evaluation command uses a separate process. Training used one A100 at a time.

## Part 1

The baseline uses `bert-base-cased`, seed 0, learning rate 5e-5, batch size 8, three epochs, a linear schedule without warmup, and maximum length 512. Augmentation adds 5,000 distinct transformed training reviews to the 25,000 originals and starts from pretrained BERT. Its three epochs therefore use more optimizer steps. Both evaluations use the same frozen transformation.

```bash
mkdir -p reproduced/part1
A4_PY1="$PWD/.venv-part1/bin/python"
(cd reproduced/part1 && "$A4_PY1" ../../part1/main.py --train --output_dir out --seed 0 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --max_length 512)
(cd reproduced/part1 && "$A4_PY1" ../../part1/main.py --eval --model_dir out)
(cd reproduced/part1 && "$A4_PY1" ../../part1/main.py --eval_transformed --model_dir out)
(cd reproduced/part1 && "$A4_PY1" ../../part1/main.py --train_augmented --output_dir out_augmented --seed 0 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --max_length 512)
(cd reproduced/part1 && "$A4_PY1" ../../part1/main.py --eval --model_dir out_augmented)
(cd reproduced/part1 && "$A4_PY1" ../../part1/main.py --eval_transformed --model_dir out_augmented)
```

The four outputs contain alternating predicted and reference binary labels, 25,000 pairs each. Measured accuracies were 93.072% and 89.372% for the baseline on original and transformed data; augmentation gave 92.788% and 90.444%. Official feedback reports these as 93.07%/89.37% and 92.79%/90.44%, respectively. Baseline original accuracy exceeds the 91% threshold, but its drop of 3.700 percentage points does not satisfy the strict greater-than-4-point criterion. Augmentation changes original accuracy by −0.284 points and transformed accuracy by +1.072 points. The transformation was not retuned on the test set; final grading remains with course staff.

## Part 2

All parameters of `google-t5/t5-small` were fine-tuned with seed 42, AdamW at 3e-4, batch size 16, linear schedule, no warmup or weight decay, at most 20 epochs and patience 5. The native tokenizer loses SQL `<` operators; the implementation adds three ordinary tokens (` <=`, ` <>`, ` <`) and saves its tokenizer with the checkpoint. Sources and targets are not truncated. Selection uses development Record F1.

```bash
.venv-part2/bin/python part2/train_t5.py --mode train --run_dir reproduced/t5_train --checkpoint_dir reproduced/t5_checkpoints --learning_rate 3e-4 --batch_size 16 --test_batch_size 16 --max_n_epochs 20 --patience_epochs 5 --seed 42 --max_new_tokens 1024 --num_beams 1
.venv-part2/bin/python part2/train_t5.py --mode dev --run_dir reproduced/t5_dev --checkpoint_dir reproduced/t5_checkpoints --test_batch_size 16 --seed 42 --max_new_tokens 1024 --num_beams 1
```

Epoch 20 was selected using development Record F1. Both the development evaluation and submitted test predictions use greedy decoding. Beam-size-three decoding was not run.

| Split | SQL query EM | Record EM | Record F1 |
| --- | ---: | ---: | ---: |
| Development | 65.021% | 80.687% | 82.385% |
| Official test | 0.00% | 74.54% | 77.41% |

The official test row comes from the reported Gradescope feedback, including SQL query EM of exactly 0.00%. Official test Record F1 of 77.41% exceeds the 65% performance threshold. The feedback lists the Q7 performance component as 25.00/25; this component is not an overall assignment grade, and final grading remains with course staff. These results do not change the selected checkpoint, decoding settings or submitted predictions.

Download all nine companion checkpoint files into one directory, such as `checkpoints/t5_final/`, from the [selected Drive checkpoint folder](https://drive.google.com/drive/folders/1BtJZwiui-ydKVWGjeVXEvkZqdeAqnmpQ). The actual `model.safetensors` SHA-256 must be `f4dbb8b3fea0f767fc934d9c3d69d0894e55e4f4af7656cfefadf88d677e61e0`. Load this directory directly, including its saved tokenizer and settings. `settings.json` retains the original training paths as provenance; the command below supplies local paths. Decoding is one beam, no sampling and at most 1,024 new tokens.

```bash
mkdir -p reproduced/t5_test
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 A4_SQL_THREADS=2 A4_SQL_TIMEOUT_SECONDS=30 .venv-part2/bin/python part2/train_t5.py --mode test --checkpoint_dir checkpoints/t5_final --run_dir reproduced/t5_test --test_batch_size 2 --seed 42 --num_beams 1 --max_new_tokens 1024 --test_sql_path reproduced/t5_test/t5_ft_experiment_test.sql --test_record_path reproduced/t5_test/t5_ft_experiment_test.pkl
```

The script automatically uses CUDA if available, otherwise CPU. Run export in a CPU-only environment to reproduce that device choice. It writes one SQL line and one record/error entry per test input in source order, retaining failed queries. The reviewed export has 432 predictions, 392 successful executions and 40 execution errors. Execution success is not semantic accuracy. Output paths must be fresh; retain the submitted predictions when reproducing them.

## Checks and saved-output evaluation

The tests use small synthetic fixtures. They verify implementation behavior, not pretrained model performance.

```bash
.venv-part1/bin/python -m unittest discover -s tests -p test_part1.py -v
.venv-part2/bin/python -m unittest discover -s tests -p test_t5.py -v
.venv-part2/bin/python -m unittest discover -s tests -p test_sql_utils.py -v
.venv-part2/bin/python part2/evaluate.py --predicted_sql reproduced/t5_dev/selected_dev/dev.sql --predicted_records reproduced/t5_dev/selected_dev/dev.pkl --development_sql part2/data/dev.sql --development_records part2/records/ground_truth_dev.pkl
```

`scripts/prepare_submission.py` collects the four BERT outputs and two T5 outputs from completed runs. Submit the written report and six prediction files separately through Gradescope; the report provides the code and selected-checkpoint links. No extra-credit model was trained.
