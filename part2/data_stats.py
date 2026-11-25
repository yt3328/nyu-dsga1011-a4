import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import transformers

from load_data import SQL_OPERATOR_TOKENS, create_t5_tokenizer, load_lines


def token_statistics(texts, tokenizer, check_roundtrips=False):
    rows = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    lengths = [len(row) for row in rows]
    vocabulary = {token for row in rows for token in row if token != tokenizer.pad_token_id}
    statistics = {
        "mean_length": float(np.mean(lengths)),
        "max_length": max(lengths),
        "length_percentiles": {str(percentile): float(np.percentile(lengths, percentile)) for percentile in [50, 90, 95, 99]},
        "observed_vocabulary_size": len(vocabulary),
        "unknown_token_count": sum(row.count(tokenizer.unk_token_id) for row in rows),
        "examples_with_unknown_tokens": sum(tokenizer.unk_token_id in row for row in rows),
    }
    if check_roundtrips:
        decoded = tokenizer.batch_decode(rows, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        matches = sum(source == prediction for source, prediction in zip(texts, decoded))
        statistics.update(exact_roundtrip_count=matches, roundtrip_mismatch_count=len(texts) - matches)
    return statistics


def collect_statistics(data_dir, tokenizer, native_tokenizer):
    statistics = {
        "schema_version": 2,
        "tokenizer": "google-t5/t5-small",
        "tokenizer_variant": "sql_operators_v1",
        "transformers_version": transformers.__version__,
        "added_regular_tokens": list(SQL_OPERATOR_TOKENS),
        "native_tokenizer_size": len(native_tokenizer),
        "extended_tokenizer_size": len(tokenizer),
        "data_sha256": {name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
                        for name in ("train.nl", "train.sql", "dev.nl", "dev.sql")},
        "definitions": {
            "special_tokens": "EOS included; PAD and decoder-start tokens excluded",
            "lengths": "Unpadded token counts with no truncation",
            "preprocessing": "Only line-ending removal; SQL strings are unchanged. The tokenizer adds three regular, space-prefixed comparison-operator tokens.",
            "before": "Original T5 tokenizer on the supplied strings after removing line endings",
            "after": "T5 tokenizer extended with the regular tokens ' <=', ' <>', and ' <' on the identical strings",
            "vocabulary": "Distinct observed token IDs in the corresponding field and split; PAD excluded",
        },
        "before": {},
        "after": {},
        "sql_roundtrip_checks": [],
    }
    for split in ["train", "dev"]:
        sources = load_lines(Path(data_dir) / f"{split}.nl")
        targets = load_lines(Path(data_dir) / f"{split}.sql")
        if not sources or len(sources) != len(targets):
            raise ValueError(f"Unaligned or empty {split} data.")
        for stage, stage_tokenizer in (("before", native_tokenizer), ("after", tokenizer)):
            statistics[stage][split] = {
                "number_of_examples": len(sources),
                "natural_language": token_statistics(sources, stage_tokenizer),
                "sql": token_statistics(targets, stage_tokenizer, check_roundtrips=True),
            }
    probes = [
        "SELECT flight_1.flight_id FROM flight flight_1 WHERE flight_1.departure_time < 1200",
        "SELECT * FROM city WHERE city_name = 'NEW YORK' AND city_code != 'BOS'",
        "SELECT COUNT(*) FROM flight WHERE departure_time >= 900 AND arrival_time <= 1700",
    ]
    for query in probes:
        ids = tokenizer(query, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        native_ids = native_tokenizer(query, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
        statistics["sql_roundtrip_checks"].append({
            "source": query,
            "native_decoded": native_tokenizer.decode(native_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            "native_unknown_token_count": native_ids.count(native_tokenizer.unk_token_id),
            "decoded": decoded,
            "exact_roundtrip": query == decoded,
            "unknown_token_count": ids.count(tokenizer.unk_token_id),
        })
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Statistics output already exists: {output}")
    tokenizer = create_t5_tokenizer()
    native_tokenizer = create_t5_tokenizer(extend_sql_operators=False)
    statistics = collect_statistics(args.data_dir, tokenizer, native_tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as output_file:
        json.dump(statistics, output_file, indent=2)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
