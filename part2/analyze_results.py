import argparse
import json
from pathlib import Path

from utils import (
    compute_metrics,
    compute_record_exact_match,
    compute_record_F1,
    load_queries_and_records,
)


def json_value(value):
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def analyze_results(args):
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f'Analysis output already exists: {output_path}')
    data_dir = Path(args.data_dir)
    with (data_dir / 'dev.nl').open(encoding='utf-8') as handle:
        inputs = [line.rstrip('\r\n') for line in handle]
    gold_queries, gold_records, gold_errors = load_queries_and_records(
        data_dir / 'dev.sql', args.ground_truth_records,
    )
    predicted_queries, predicted_records, predicted_errors = load_queries_and_records(
        args.predicted_sql, args.predicted_records,
    )
    counts = {
        'inputs': len(inputs),
        'reference_queries': len(gold_queries),
        'reference_records': len(gold_records),
        'reference_errors': len(gold_errors),
        'predicted_queries': len(predicted_queries),
        'predicted_records': len(predicted_records),
        'predicted_errors': len(predicted_errors),
    }
    if not inputs or len(set(counts.values())) != 1:
        raise ValueError(f'Development evidence counts differ or are empty: {counts}')
    sql_em, record_em, record_f1, _ = compute_metrics(
        data_dir / 'dev.sql', args.predicted_sql,
        args.ground_truth_records, args.predicted_records,
    )
    diagnostics = {
        'timeout': 0,
        'invalid_sql_or_execution_failure': 0,
        'valid_records_mismatch': 0,
        'sql_differs_but_records_match': 0,
        'sql_and_records_match': 0,
    }
    examples = []
    for index in range(len(inputs)):
        error = predicted_errors[index]
        sql_identical = gold_queries[index] == predicted_queries[index]
        record_exact = bool(compute_record_exact_match(
            [gold_records[index]], [predicted_records[index]],
        ))
        if error:
            category = 'timeout' if 'timed out' in error.lower() or 'timeout' in error.lower() else 'invalid_sql_or_execution_failure'
        elif not record_exact:
            category = 'valid_records_mismatch'
        elif not sql_identical:
            category = 'sql_differs_but_records_match'
        else:
            category = 'sql_and_records_match'
        diagnostics[category] += 1
        examples.append({
            'index': index,
            'input': inputs[index],
            'reference_sql': gold_queries[index],
            'predicted_sql': predicted_queries[index],
            'reference_error': gold_errors[index],
            'predicted_error': error,
            'sql_identical': sql_identical,
            'record_exact': record_exact,
            'record_f1': float(compute_record_F1(
                [gold_records[index]], [predicted_records[index]],
            )),
            'diagnostic_category': category,
            'reference_record_count': len(gold_records[index]),
            'predicted_record_count': len(predicted_records[index]),
            'reference_record_sample': gold_records[index][:20],
            'predicted_record_sample': predicted_records[index][:20],
        })
    result = {
        'scope': 'Full supplied development set, in source order; no sampling of examples.',
        'metric_scale': 'Fractions in [0, 1] using the unchanged instructor metric functions.',
        'counts': counts,
        'metrics': {'sql_em': float(sql_em), 'record_em': float(record_em), 'record_f1': float(record_f1)},
        'execution': {
            'valid': sum(not bool(error) for error in predicted_errors),
            'failed_including_timeouts': sum(bool(error) for error in predicted_errors),
            'timeouts': diagnostics['timeout'],
        },
        'diagnostic_counts': diagnostics,
        'diagnostic_denominator': len(inputs),
        'diagnostic_note': 'These execution and record-comparison groups are diagnostics, not a substitute for inspecting at least three genuine semantic error classes.',
        'record_sample_note': 'Each record sample contains the first 20 rows at most, in saved order. Full row counts are reported separately. Metric functions compare record sets; duplicates are retained in these samples.',
        'examples': examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=json_value)
        handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_sql', required=True)
    parser.add_argument('--predicted_records', required=True)
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parent / 'data'))
    parser.add_argument('--ground_truth_records', default=str(Path(__file__).resolve().parent / 'records' / 'ground_truth_dev.pkl'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze_results(args)
    print(json.dumps({'metrics': result['metrics'], 'diagnostic_counts': result['diagnostic_counts'], 'count': result['diagnostic_denominator']}, indent=2))


if __name__ == '__main__':
    main()
