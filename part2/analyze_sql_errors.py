"""Read saved dev artifacts and suggest evidence-backed SQL error groups.

The structural groups overlap and identify differences to inspect, not proven
causes. No SQL is rewritten or executed, and no model predictions are created.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from load_data import load_lines
from utils import (compute_metrics, compute_record_exact_match, compute_record_F1,
                   load_queries_and_records)


TOKEN = re.compile(
    r"--[^\r\n]*|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|"
    r"`(?:``|[^`])*`|\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_$]*|"
    r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|<=|>=|<>|!=|==|[^\s]", re.S)
IDENTIFIER = re.compile(r"^[A-Z_][A-Z0-9_$]*$")
NUMBER = re.compile(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
CLAUSE_END = {"WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "UNION", "EXCEPT", "INTERSECT", ";"}
RESERVED = CLAUSE_END | {"SELECT", "FROM", "JOIN", "ON", "AS", "INNER", "LEFT", "RIGHT", "FULL",
                         "OUTER", "CROSS", "NATURAL", "AND", "OR", "NOT", "USING", "OFFSET"}
CATEGORIES = {
    "execution_failure": "The saved SQL execution error is nonempty, including timeouts.",
    "entity_or_literal_difference": "Executable SQL returns wrong records and contains different quoted strings or numeric constants.",
    "table_or_join_difference": "Executable SQL returns wrong records and differs in table occurrences or column-equality join predicates.",
    "projection_or_aggregation_difference": "Executable SQL returns wrong records and differs in SELECT expressions, DISTINCT, or aggregate functions.",
    "condition_or_comparison_difference": "Executable SQL returns wrong records and differs in simple comparison operands/operators.",
    "logic_or_nested_query_difference": "Executable SQL returns wrong records and differs in Boolean/operator sequence, SELECT nesting, GROUP BY, ORDER BY, or LIMIT.",
    "unclassified_executable_record_error": "Executable SQL returns wrong records but none of the structural signatures identifies a difference.",
}


def tokens_for(query):
    tokens = []
    for match in TOKEN.finditer(query):
        token = match.group()
        if token.startswith(("--", "/*")):
            continue
        tokens.append(token if token.startswith("'") else token.upper())
    return tokens


def counted(items):
    return dict(sorted(Counter(items).items()))


def sql_features(query):
    """Extract conservative lexical structure, with aliases resolved per SELECT.

    This is not a complete SQL parser. In particular, derived-table aliases,
    CTEs, and arbitrary expressions are retained lexically rather than given
    semantic equivalence rules. The supplied ATIS-style aliases are supported.
    """
    tokens = tokens_for(query)
    depths, scopes, parents, stack = [], [], {}, []
    depth = 0
    next_scope = 0
    for token in tokens:
        if token == "SELECT":
            while stack and stack[-1][0] >= depth:
                stack.pop()
            parents[next_scope] = stack[-1][1] if stack else None
            stack.append((depth, next_scope))
            next_scope += 1
        depths.append(depth)
        scopes.append(stack[-1][1] if stack else None)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            while stack and stack[-1][0] > depth:
                stack.pop()

    aliases, table_counts = defaultdict(dict), defaultdict(Counter)
    tables = []
    for index, token in enumerate(tokens):
        if token != "FROM":
            continue
        scope, level = scopes[index], depths[index]
        expecting_table = True
        cursor = index + 1
        while cursor < len(tokens):
            current = tokens[cursor]
            if depths[cursor] < level:
                break
            if depths[cursor] != level or scopes[cursor] != scope:
                cursor += 1
                continue
            if current in CLAUSE_END:
                break
            if current in {",", "JOIN"}:
                expecting_table = True
            elif expecting_table and IDENTIFIER.fullmatch(current) and current not in RESERVED:
                table = current
                end = cursor + 1
                if end + 1 < len(tokens) and tokens[end] == "." and IDENTIFIER.fullmatch(tokens[end + 1]):
                    table += "." + tokens[end + 1]
                    end += 2
                alias_at = end + int(end < len(tokens) and tokens[end] == "AS")
                alias = table
                if alias_at < len(tokens) and IDENTIFIER.fullmatch(tokens[alias_at]) and tokens[alias_at] not in RESERVED:
                    alias = tokens[alias_at]
                table_counts[scope][table] += 1
                role = f"S{scope}:{table}#{table_counts[scope][table]}"
                aliases[scope][alias] = role
                aliases[scope].setdefault(table, role)
                tables.append(table)
                expecting_table = False
                cursor = end - 1
            cursor += 1

    def resolve(alias, scope):
        while scope is not None:
            if alias in aliases[scope]:
                return aliases[scope][alias]
            scope = parents.get(scope)
        return alias

    canonical, canonical_scopes, canonical_depths = [], [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (index + 2 < len(tokens) and IDENTIFIER.fullmatch(token) and tokens[index + 1] == "."
                and (IDENTIFIER.fullmatch(tokens[index + 2]) or tokens[index + 2] == "*")):
            canonical.append("@" + resolve(token, scopes[index]) + "." + tokens[index + 2])
            canonical_scopes.append(scopes[index])
            canonical_depths.append(depths[index])
            index += 3
        else:
            canonical.append(token)
            canonical_scopes.append(scopes[index])
            canonical_depths.append(depths[index])
            index += 1

    projections, comparisons, joins = [], [], []
    for index, token in enumerate(canonical):
        if token == "SELECT":
            end = index + 1
            while end < len(canonical):
                if (canonical[end] == "FROM" and canonical_scopes[end] == canonical_scopes[index]
                        and canonical_depths[end] == canonical_depths[index]):
                    break
                end += 1
            projections.append(" ".join(canonical[index + 1:end]))
        if token in {"=", "==", "!=", "<>", "<", "<=", ">", ">="} and 0 < index < len(canonical) - 1:
            left, right = canonical[index - 1], canonical[index + 1]
            # Parenthesized expressions need manual review; avoid pretending that
            # their closing/opening delimiters identify comparison operands.
            if left not in {"(", ")"} and right not in {"(", ")"}:
                comparisons.append(f"{left} {token} {right}")
            if token in {"=", "=="} and left.startswith("@") and right.startswith("@"):
                joins.append(" = ".join(sorted((left, right))))
    literals = [token for token in canonical if token.startswith("'") or NUMBER.fullmatch(token)]
    aggregates = [token for index, token in enumerate(canonical[:-1])
                  if token in {"COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT"} and canonical[index + 1] == "("]
    logic = [token for token in canonical if token in {"AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "EXISTS", "IS", "UNION", "EXCEPT", "INTERSECT"}]
    trailing_clauses = []
    for index, token in enumerate(canonical):
        if token in {"GROUP", "ORDER", "LIMIT", "HAVING"}:
            end = index + 1
            while end < len(canonical) and canonical[end] not in CLAUSE_END:
                end += 1
            trailing_clauses.append(" ".join(canonical[index:end]))
    return {"tables": counted(tables), "joins": counted(joins), "projections": projections,
            "aggregates": counted(aggregates), "literals": counted(literals),
            "comparisons": counted(comparisons), "logical_operators": logic,
            "select_depths": [depths[i] for i, token in enumerate(tokens) if token == "SELECT"],
            "group_order_limit_having": trailing_clauses}


def changed_features(reference, prediction):
    return {key: {"reference": reference[key], "prediction": prediction[key]}
            for key in reference if reference[key] != prediction[key]}


def semantic_tags(reference, prediction):
    changed = changed_features(reference, prediction)
    groups = []
    for category, features in (
        ("entity_or_literal_difference", ("literals",)),
        ("table_or_join_difference", ("tables", "joins")),
        ("projection_or_aggregation_difference", ("projections", "aggregates")),
        ("condition_or_comparison_difference", ("comparisons",)),
        ("logic_or_nested_query_difference", ("logical_operators", "select_depths", "group_order_limit_having")),
    ):
        if any(feature in changed for feature in features):
            groups.append(category)
    return groups or ["unclassified_executable_record_error"], changed


def analyze(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Preserve the existing analysis: {output}")
    data = Path(args.data_dir)
    inputs = load_lines(data / "dev.nl")
    references, reference_records, reference_errors = load_queries_and_records(data / "dev.sql", args.ground_truth_records)
    predictions, predicted_records, predicted_errors = load_queries_and_records(args.predicted_sql, args.predicted_records)
    lengths = [len(values) for values in (inputs, references, reference_records, reference_errors,
                                         predictions, predicted_records, predicted_errors)]
    if not inputs or len(set(lengths)) != 1:
        raise ValueError(f"Nonempty aligned development artifacts are required: {lengths}")
    sql_em, record_em, record_f1, _ = compute_metrics(data / "dev.sql", args.predicted_sql,
                                                    args.ground_truth_records, args.predicted_records)
    counts = Counter({category: 0 for category in CATEGORIES})
    broad = Counter({name: 0 for name in ("timeout", "invalid_sql_or_execution_failure",
                                        "valid_records_mismatch", "sql_differs_but_records_match",
                                        "sql_and_records_match")})
    grouped = defaultdict(list)
    rows = []
    for index, (gold, predicted) in enumerate(zip(references, predictions)):
        exact = bool(compute_record_exact_match([reference_records[index]], [predicted_records[index]]))
        f1 = float(compute_record_F1([reference_records[index]], [predicted_records[index]]))
        groups, changes = [], {}
        if predicted_errors[index]:
            groups = ["execution_failure"]
            error_text = str(predicted_errors[index]).lower()
            broad["timeout" if "timed out" in error_text or "timeout" in error_text
                  else "invalid_sql_or_execution_failure"] += 1
        elif not exact:
            groups, changes = semantic_tags(sql_features(gold), sql_features(predicted))
            broad["valid_records_mismatch"] += 1
        else:
            broad["sql_and_records_match" if gold == predicted else "sql_differs_but_records_match"] += 1
        row = {"index": index, "input": inputs[index], "reference_sql": gold, "predicted_sql": predicted,
               "reference_error": reference_errors[index], "predicted_error": predicted_errors[index],
               "record_exact": exact, "record_f1": f1,
               "reference_record_count": len(reference_records[index]),
               "predicted_record_count": len(predicted_records[index]),
               "candidate_categories": groups, "structural_differences": changes}
        rows.append(row)
        for category in groups:
            counts[category] += 1
            grouped[category].append(row)
    category_results = {}
    for category, definition in CATEGORIES.items():
        examples = sorted(grouped[category], key=lambda row: (row["record_f1"], row["index"]))
        category_results[category] = {"definition": definition, "count": counts[category],
                                      "denominator": len(inputs), "fraction": counts[category] / len(inputs),
                                      "examples": examples[:args.examples_per_category]}
    result = {"scope": "Full supplied development set in original order, using saved predictions and database records.",
              "metric_scale": "Fractions; unchanged instructor metrics.",
              "metrics": {"sql_em": float(sql_em), "record_em": float(record_em), "record_f1": float(record_f1)},
              "counts": {"examples": len(inputs), "reference_execution_failures": sum(bool(e) for e in reference_errors),
                         "predicted_execution_failures": sum(bool(e) for e in predicted_errors),
                         "executable_record_errors": sum(not row["record_exact"] and not row["predicted_error"] for row in rows),
                         "record_exact": sum(row["record_exact"] for row in rows)},
              "category_note": "Candidate categories overlap; semantic tags apply only to executable predictions with mismatched records. Inspect the real paired SQL before treating a structural difference as the cause of an error. Execution failures are reported even when empty result sets happen to match.",
              "broad_diagnostic_counts": dict(broad),
              "broad_diagnostic_denominator": len(inputs),
              "limitations": ["This lexical structural analyzer is not a full SQL parser or semantic-equivalence checker.",
                              "Alias names are resolved by SELECT scope; repeated tables use FROM occurrence order, so equivalent join reorderings can alter signatures.",
                              "Derived tables, CTEs, arbitrary expression equivalences and correlated expressions can require manual interpretation.",
                              "Literal differences include numeric constants as well as entities; an observed difference is not automatically an entity-resolution mistake.",
                              "Reference execution errors, if any, must be reviewed separately."],
              "categories": category_results, "examples": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted_sql", required=True)
    parser.add_argument("--predicted_records", required=True)
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--ground_truth_records", default=str(Path(__file__).resolve().parent / "records/ground_truth_dev.pkl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples_per_category", type=int, default=3)
    args = parser.parse_args()
    if args.examples_per_category < 1:
        parser.error("--examples_per_category must be positive")
    result = analyze(args)
    print(json.dumps({"metrics": result["metrics"], "counts": result["counts"],
                      "category_counts": {name: entry["count"] for name, entry in result["categories"].items()}}, indent=2))


if __name__ == "__main__":
    main()
