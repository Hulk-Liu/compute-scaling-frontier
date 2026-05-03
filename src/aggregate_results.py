"""Aggregate raw per-example outputs into one experiment-cell summary row."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from src.cost_accounting import (
    DEFAULT_PRICES_PATH,
    inference_cost_per_query_usd,
    load_prices,
    total_serving_cost_usd,
    training_cost_usd,
)
from src.evaluator import build_answer_diagnostics, evaluate_batch


DEFAULT_QUERY_COUNTS = (1_000, 10_000, 100_000, 1_000_000)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into dictionaries."""

    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object")
            rows.append(row)
    return rows


def _prediction_and_gold(row: dict[str, Any], row_index: int) -> tuple[str, str]:
    try:
        prediction = row["prediction"]
    except KeyError as e:
        raise ValueError(f"Row {row_index} is missing required field 'prediction'") from e

    gold = row.get("gold_answer", row.get("answer"))
    if gold is None:
        raise ValueError(
            f"Row {row_index} must contain either 'gold_answer' or 'answer'"
        )
    if not isinstance(prediction, str) or not isinstance(gold, str):
        raise ValueError(f"Row {row_index} prediction and gold fields must be strings")
    return prediction, gold


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def aggregate_answer_diagnostics(
    prediction_gold_pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Aggregate prompt-compliance diagnostics for one experiment cell."""

    diagnostics = [
        build_answer_diagnostics(prediction, gold)
        for prediction, gold in prediction_gold_pairs
    ]
    total = len(diagnostics)
    final_marker_count = sum(
        1 for diagnostic in diagnostics if diagnostic["has_final_marker"]
    )
    answer_format_ok_count = sum(
        1 for diagnostic in diagnostics if diagnostic["answer_format_ok"]
    )

    return {
        "has_final_marker_count": final_marker_count,
        "has_final_marker_rate": _rate(final_marker_count, total),
        "answer_format_ok_count": answer_format_ok_count,
        "answer_format_ok_rate": _rate(answer_format_ok_count, total),
        "missing_final_marker_count": total - final_marker_count,
        "malformed_final_marker_count": final_marker_count - answer_format_ok_count,
    }


def aggregate_cell(
    raw_rows: Iterable[dict[str, Any]],
    prices: dict[str, Any],
    train_size: int,
    strategy: str,
    budget: int,
    train_gpu_hours: float,
    model_tokens_per_sample: int,
    judge_model: str | None = None,
    judge_input_tokens: int = 0,
    judge_output_tokens: int = 0,
    query_counts: Iterable[int] = DEFAULT_QUERY_COUNTS,
) -> dict[str, Any]:
    """Aggregate one train-size/strategy/budget cell."""

    materialized = list(raw_rows)
    pairs = [
        _prediction_and_gold(row, row_index=i)
        for i, row in enumerate(materialized, start=1)
    ]
    predictions = [prediction for prediction, _gold in pairs]
    gold_answers = [gold for _prediction, gold in pairs]
    _records, eval_summary = evaluate_batch(predictions, gold_answers)
    answer_diagnostics = aggregate_answer_diagnostics(pairs)

    train_cost = training_cost_usd(
        prices,
        synthetic_examples=train_size,
        gpu_hours=train_gpu_hours,
    )
    inference_cost = inference_cost_per_query_usd(
        prices,
        budget=budget,
        model_tokens_per_sample=model_tokens_per_sample,
        judge_model=judge_model,
        judge_input_tokens=judge_input_tokens,
        judge_output_tokens=judge_output_tokens,
    )

    row: dict[str, Any] = {
        "train_size": train_size,
        "strategy": strategy,
        "budget": budget,
        "n_eval": eval_summary.total,
        "correct": eval_summary.correct,
        "accuracy": eval_summary.accuracy,
        **answer_diagnostics,
        "train_gpu_hours": train_gpu_hours,
        "train_cost_usd": train_cost.total_usd,
        "model_tokens_per_sample": model_tokens_per_sample,
        "judge_model": judge_model or "",
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
        "inference_cost_per_query_usd": inference_cost.total_usd,
    }

    for query_count in query_counts:
        row[f"total_cost_usd_at_{query_count}"] = total_serving_cost_usd(
            train_cost_usd=train_cost.total_usd,
            inference_cost_per_query_usd=inference_cost.total_usd,
            query_count=query_count,
        )

    return row


def write_aggregated_csv(rows: list[dict[str, Any]], output_path: Path | str) -> None:
    """Write aggregate rows to CSV."""

    if not rows:
        raise ValueError("rows cannot be empty")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_jsonl", type=Path)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--train-gpu-hours", type=float, default=0.0)
    parser.add_argument("--model-tokens-per-sample", type=int, default=0)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-input-tokens", type=int, default=0)
    parser.add_argument("--judge-output-tokens", type=int, default=0)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/aggregated.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prices = load_prices(args.prices)
    row = aggregate_cell(
        raw_rows=read_jsonl(args.raw_jsonl),
        prices=prices,
        train_size=args.train_size,
        strategy=args.strategy,
        budget=args.budget,
        train_gpu_hours=args.train_gpu_hours,
        model_tokens_per_sample=args.model_tokens_per_sample,
        judge_model=args.judge_model,
        judge_input_tokens=args.judge_input_tokens,
        judge_output_tokens=args.judge_output_tokens,
    )
    write_aggregated_csv([row], args.output)
    print(f"Wrote 1 aggregate row to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
