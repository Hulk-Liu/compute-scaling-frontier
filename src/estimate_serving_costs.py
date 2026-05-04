"""Estimate serving-token costs from raw Qwen eval outputs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from src.aggregate_results import (
    DEFAULT_QUERY_COUNTS,
    read_jsonl,
    write_aggregated_csv,
)
from src.cost_accounting import (
    DEFAULT_PRICES_PATH,
    inference_cost_per_query_usd,
    load_prices,
    total_serving_cost_usd,
)
from src.run_its_experiment import build_math_prompt, _response_content


DEFAULT_AGGREGATE_PATH = Path("results/aggregated.csv")
DEFAULT_RAW_DIR = Path("results/raw")
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class TokenCounter(Protocol):
    """Interface for token counting implementations."""

    method_name: str

    def count(self, text: str) -> int:
        """Count tokens for a text string."""


@dataclass(frozen=True)
class CharacterHeuristicCounter:
    """Approximate tokens using a simple characters-per-token heuristic."""

    chars_per_token: float = 4.0
    method_name: str = "char_heuristic_4"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / self.chars_per_token))


class HuggingFaceTokenCounter:
    """Count tokens with a Hugging Face tokenizer."""

    method_name = "hf_tokenizer"

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def build_token_counter(
    method: str,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
) -> TokenCounter:
    """Build a token counter by method name."""

    if method == "char":
        return CharacterHeuristicCounter()
    if method == "hf":
        return HuggingFaceTokenCounter(tokenizer_model)
    if method == "auto":
        try:
            return HuggingFaceTokenCounter(tokenizer_model)
        except Exception as exc:
            print(
                f"Falling back to character token heuristic because "
                f"{tokenizer_model!r} tokenizer could not be loaded: {exc}"
            )
            return CharacterHeuristicCounter()
    raise ValueError(f"Unsupported token count method: {method}")


def read_aggregate_rows(path: Path | str) -> list[dict[str, Any]]:
    """Read aggregate CSV rows without coercing or dropping columns."""

    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No aggregate rows found in {path}")
    return rows


def raw_path_for_row(row: dict[str, Any], raw_dir: Path) -> Path:
    """Resolve the raw JSONL path for an aggregate row."""

    train_size = int(row["train_size"])
    strategy = row["strategy"]
    budget = int(row["budget"])

    if train_size == 0:
        model_slug = "base"
    elif train_size == 100:
        model_slug = "lora_n100"
    elif train_size == 500:
        model_slug = "lora_n500"
    else:
        model_slug = f"lora_n{train_size}"

    if strategy == "greedy":
        strategy_slug = "greedy"
    else:
        strategy_slug = f"{strategy}{budget}"

    return raw_dir / f"qwen_{model_slug}_{strategy_slug}.jsonl"


def candidate_responses(raw_row: dict[str, Any]) -> list[str]:
    """Return all generated candidate response contents for one raw row."""

    all_responses = raw_row.get("all_responses")
    if isinstance(all_responses, list) and all_responses:
        return [
            _response_content(response)
            for response in all_responses
            if isinstance(response, dict)
        ]
    return [str(raw_row.get("prediction", ""))]


def estimate_model_tokens_per_sample(
    raw_rows: list[dict[str, Any]],
    counter: TokenCounter,
) -> int:
    """Estimate average prompt+completion tokens for one generated sample."""

    sample_token_counts: list[int] = []
    for raw_row in raw_rows:
        prompt = build_math_prompt(str(raw_row["question"]))
        prompt_tokens = counter.count(prompt)
        for response in candidate_responses(raw_row):
            sample_token_counts.append(prompt_tokens + counter.count(response))

    if not sample_token_counts:
        raise ValueError("Cannot estimate tokens from empty raw rows")
    return int(round(mean(sample_token_counts)))


def update_row_costs(
    row: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    prices: dict[str, Any],
    counter: TokenCounter,
    query_counts: tuple[int, ...] = DEFAULT_QUERY_COUNTS,
) -> dict[str, Any]:
    """Return an aggregate row with serving-token cost fields updated."""

    updated = dict(row)
    budget = int(row["budget"])
    train_cost = float(row["train_cost_usd"])
    tokens_per_sample = estimate_model_tokens_per_sample(raw_rows, counter)
    inference_cost = inference_cost_per_query_usd(
        prices,
        budget=budget,
        model_tokens_per_sample=tokens_per_sample,
        judge_model=row.get("judge_model") or None,
        judge_input_tokens=int(row.get("judge_input_tokens") or 0),
        judge_output_tokens=int(row.get("judge_output_tokens") or 0),
    )

    updated["model_tokens_per_sample"] = tokens_per_sample
    updated["token_estimation_method"] = counter.method_name
    updated["inference_cost_per_query_usd"] = round(inference_cost.total_usd, 8)
    for query_count in query_counts:
        total_cost = total_serving_cost_usd(
            train_cost_usd=train_cost,
            inference_cost_per_query_usd=inference_cost.total_usd,
            query_count=query_count,
        )
        updated[f"total_cost_usd_at_{query_count}"] = round(total_cost, 6)
    return updated


def update_aggregate_costs(
    aggregate_path: Path,
    raw_dir: Path,
    prices_path: Path,
    output_path: Path,
    counter: TokenCounter,
) -> list[dict[str, Any]]:
    """Update aggregate CSV cost columns from raw model outputs."""

    prices = load_prices(prices_path)
    updated_rows = []
    for row in read_aggregate_rows(aggregate_path):
        raw_path = raw_path_for_row(row, raw_dir)
        raw_rows = read_jsonl(raw_path)
        updated_rows.append(
            update_row_costs(
                row=row,
                raw_rows=raw_rows,
                prices=prices,
                counter=counter,
            )
        )

    write_aggregated_csv(updated_rows, output_path)
    return updated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_AGGREGATE_PATH)
    parser.add_argument(
        "--token-method",
        choices=["auto", "hf", "char"],
        default="auto",
        help="Use HF tokenizer when available, or char heuristic with auto fallback.",
    )
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counter = build_token_counter(args.token_method, args.tokenizer_model)
    rows = update_aggregate_costs(
        aggregate_path=args.aggregate,
        raw_dir=args.raw_dir,
        prices_path=args.prices,
        output_path=args.output,
        counter=counter,
    )
    print(
        f"Wrote {len(rows)} rows with serving-token cost estimates to {args.output} "
        f"using {counter.method_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
