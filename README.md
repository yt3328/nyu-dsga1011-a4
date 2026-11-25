# DS-GA 1011 Assignment 4

This project fine-tunes BERT for sentiment classification and T5 for text-to-SQL. It examines how a fixed keyboard-typo transformation affects sentiment predictions and whether training augmentation reduces that sensitivity. The implementation builds on the [course starter repository](https://github.com/gregdurrett/nyu-dsga1011-a4); original instructions and attribution remain in [STARTER_README.md](STARTER_README.md) and the two part directories.

`part1/` contains BERT training, evaluation and the transformation. `part2/` contains T5 training, SQL generation, database evaluation and error analysis. The flight inputs, SQL references, database/schema and development records are included. Model checkpoints and generated predictions are distributed separately.

## Setup

Use separate Python 3.11 environments because the two parts require incompatible Transformers versions. Run setup from the repository root:

```bash
python3.11 -m venv .venv-part1
.venv-part1/bin/python -m pip install -r part1/requirements.txt
python3.11 -m venv .venv-part2
.venv-part2/bin/python -m pip install -r part2/requirements.txt
```

## BERT results

Both models use `bert-base-cased` and three training epochs. The augmented model starts from pretrained BERT and adds 5,000 transformed training reviews to the 25,000 originals. The same frozen transformation is used for both evaluations; it was not retuned on the test set.

| Model | Original test accuracy | Transformed test accuracy |
| --- | ---: | ---: |
| Baseline | 93.072% | 89.372% |
| Augmented | 92.788% | 90.444% |

The baseline drop is 3.700 percentage points, below the assignment's strict greater-than-4-point criterion. Augmentation changes original accuracy by −0.284 points and transformed accuracy by +1.072 points, with more optimizer steps because its training set is larger.

## T5 development results and checkpoint

All parameters of `google-t5/t5-small` were fine-tuned. The saved tokenizer preserves SQL comparison operators. Epoch 20 was selected by development Record F1 using greedy decoding: **SQL EM 65.021%, Record EM 80.687%, Record F1 82.385%**. Beam-size-three decoding was not run. Official hidden-test metrics are unavailable; development scores and SQL execution success do not establish test accuracy.

Download all nine files from the [selected T5 checkpoint folder](https://drive.google.com/drive/folders/1BtJZwiui-ydKVWGjeVXEvkZqdeAqnmpQ) into one directory and load its saved tokenizer with the model.

See [detailed reproduction instructions](docs/reproduction.md) for training, evaluation, tests, CPU export, the checkpoint hash and platform notes. Use fresh output paths when reproducing the completed experiments.
