"""Validate generated SFT records against their GSM8K gold answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluator import EvaluationRecord, evaluate_answer, summarize


def read_training_records(path: Path | str) -> list[dict[str, Any]]:
    """Read generated training records from JSONL."""

    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from e
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object")
            records.append(record)
    return records


def validate_record(record: dict[str, Any], row_index: int) -> EvaluationRecord:
    """Validate one generated record."""

    teacher_response = record.get("teacher_response")
    gold_answer = record.get("gold_answer")
    if not isinstance(teacher_response, str):
        raise ValueError(f"Row {row_index} is missing string field 'teacher_response'")
    if not isinstance(gold_answer, str):
        raise ValueError(f"Row {row_index} is missing string field 'gold_answer'")
    return evaluate_answer(teacher_response, gold_answer)


def validate_records(records: list[dict[str, Any]]) -> tuple[list[EvaluationRecord], float]:
    """Validate generated records and return per-row records plus accuracy."""

    evaluations = [
        validate_record(record, row_index=i)
        for i, record in enumerate(records, start=1)
    ]
    summary = summarize(evaluations)
    return evaluations, summary.accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero if any teacher answer does not match the gold answer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_training_records(args.jsonl)
    evaluations, accuracy = validate_records(records)
    correct = sum(evaluation.is_correct for evaluation in evaluations)
    total = len(evaluations)
    print(f"Validated {total} records: correct={correct}, accuracy={accuracy:.3f}")

    for index, evaluation in enumerate(evaluations, start=1):
        if not evaluation.is_correct:
            print(
                f"Mismatch row {index}: prediction={evaluation.extracted_prediction!r}, "
                f"gold={evaluation.extracted_gold!r}"
            )

    if args.fail_on_mismatch and correct != total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
