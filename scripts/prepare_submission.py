import argparse
import json
import pickle
import shutil
from pathlib import Path


def read_sentiment_predictions(path, expected_examples):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 * expected_examples or any(line not in {"0", "1"} for line in lines):
        raise ValueError(f"{path} must contain {expected_examples} alternating binary prediction/reference pairs.")
    predictions = [int(value) for value in lines[::2]]
    references = [int(value) for value in lines[1::2]]
    accuracy = sum(pred == label for pred, label in zip(predictions, references)) / expected_examples
    return accuracy, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_run", required=True, type=Path)
    parser.add_argument("--augmented_run", required=True, type=Path)
    parser.add_argument("--test_sql", required=True, type=Path)
    parser.add_argument("--test_records", required=True, type=Path)
    parser.add_argument("--test_inputs", type=Path,
                        default=Path(__file__).resolve().parents[1] / "part2/data/test.nl")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected_imdb_examples", type=int, default=25000)
    args = parser.parse_args()
    if args.expected_imdb_examples < 1:
        parser.error("expected_imdb_examples must be positive.")
    files = [
        (args.baseline_run / "out_original.txt", "out_original.txt"),
        (args.baseline_run / "out_transformed.txt", "out_transformed.txt"),
        (args.augmented_run / "out_augmented_original.txt", "out_augmented_original.txt"),
        (args.augmented_run / "out_augmented_transformed.txt", "out_augmented_transformed.txt"),
    ]
    accuracies = {}
    expected_references = None
    for path, filename in files:
        accuracy, references = read_sentiment_predictions(path, args.expected_imdb_examples)
        if expected_references is not None and references != expected_references:
            raise ValueError(f"Test reference order differs at {path}.")
        expected_references = references
        accuracies[filename] = accuracy
    with open(args.test_records, "rb") as record_file:
        data = pickle.load(record_file)
    if not isinstance(data, tuple) or len(data) != 2:
        raise ValueError("T5 pickle must contain (records, error_msgs).")
    records, errors = data
    query_count = len(args.test_sql.read_text(encoding="utf-8").splitlines())
    input_count = len(args.test_inputs.read_text(encoding="utf-8").splitlines())
    if input_count == 0 or not query_count == input_count == len(records) == len(errors):
        raise ValueError("Test input, SQL, record, and error counts differ.")
    files.extend([(args.test_sql, "t5_ft_experiment_test.sql"),
                  (args.test_records, "t5_ft_experiment_test.pkl")])
    args.destination.mkdir(parents=True, exist_ok=False)
    for source, filename in files:
        shutil.copy2(source, args.destination / filename)
    print(json.dumps({
        "sentiment_accuracies_fraction": accuracies,
        "baseline_drop_percentage_points": 100 * (accuracies["out_original.txt"] - accuracies["out_transformed.txt"]),
        "test_query_count": query_count,
        "test_execution_errors": sum(bool(error) for error in errors),
        "official_test_scores": "Pending official feedback",
        "submission_directory": str(args.destination.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
