"""Filter generated SFT records by GSM8K final-answer correctness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.prepare_eval_set import write_jsonl
from src.validate_training_data import read_training_records, validate_record


@dataclass(frozen=True)
class FilterSummary:
    """Summary of generated-record filtering."""

    total: int
    valid: int
    invalid: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.valid / self.total


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    """Return default valid/invalid JSONL paths for an input JSONL file."""

    return (
        input_path.with_name(f"{input_path.stem}_valid{input_path.suffix}"),
        input_path.with_name(f"{input_path.stem}_invalid{input_path.suffix}"),
    )


def validation_metadata(row_index: int, evaluation) -> dict[str, Any]:
    """Build serializable validation metadata for an invalid record."""

    return {
        "row_index": row_index,
        "extracted_prediction": evaluation.extracted_prediction,
        "extracted_gold": evaluation.extracted_gold,
        "is_correct": evaluation.is_correct,
    }


def split_valid_invalid(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], FilterSummary]:
    """Split records into valid and invalid lists."""

    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for row_index, record in enumerate(records, start=1):
        evaluation = validate_record(record, row_index=row_index)
        if evaluation.is_correct:
            valid_records.append(record)
        else:
            invalid_records.append(
                {
                    **record,
                    "validation": validation_metadata(row_index, evaluation),
                }
            )

    summary = FilterSummary(
        total=len(records),
        valid=len(valid_records),
        invalid=len(invalid_records),
    )
    return valid_records, invalid_records, summary


def filter_training_file(
    input_path: Path | str,
    valid_output: Path | str | None = None,
    invalid_output: Path | str | None = None,
) -> FilterSummary:
    """Filter a generated training file and write valid/invalid JSONL outputs."""

    input_path = Path(input_path)
    default_valid, default_invalid = default_output_paths(input_path)
    valid_output = Path(valid_output) if valid_output is not None else default_valid
    invalid_output = (
        Path(invalid_output) if invalid_output is not None else default_invalid
    )

    records = read_training_records(input_path)
    valid_records, invalid_records, summary = split_valid_invalid(records)
    write_jsonl(valid_records, valid_output)
    write_jsonl(invalid_records, invalid_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--valid-output", type=Path)
    parser.add_argument("--invalid-output", type=Path)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Exit nonzero if filtered accuracy is below this threshold.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    valid_output, invalid_output = default_output_paths(args.jsonl)
    if args.valid_output is not None:
        valid_output = args.valid_output
    if args.invalid_output is not None:
        invalid_output = args.invalid_output

    summary = filter_training_file(
        input_path=args.jsonl,
        valid_output=valid_output,
        invalid_output=invalid_output,
    )
    print(
        f"Filtered {summary.total} records: valid={summary.valid}, "
        f"invalid={summary.invalid}, accuracy={summary.accuracy:.3f}"
    )
    print(f"Wrote valid records to {valid_output}")
    print(f"Wrote invalid records to {invalid_output}")

    if summary.accuracy < args.min_accuracy:
        print(
            f"Accuracy {summary.accuracy:.3f} is below "
            f"minimum {args.min_accuracy:.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
