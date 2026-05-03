"""Prepare the fixed GSM8K evaluation subset used by this project."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DATASET_NAME = "gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_SPLIT = "test"
DEFAULT_SEED = 20260503
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_OUTPUT_PATH = Path("data/eval_gsm8k_50.jsonl")


def sample_indices(population_size: int, sample_size: int, seed: int) -> list[int]:
    """Return a deterministic sorted sample of dataset row indices."""

    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > population_size:
        raise ValueError(
            f"sample_size ({sample_size}) cannot exceed population_size ({population_size})"
        )

    rng = random.Random(seed)
    return sorted(rng.sample(range(population_size), sample_size))


def normalize_record(row: dict, source_index: int) -> dict:
    """Keep only the fields the downstream evaluator needs."""

    return {
        "id": f"gsm8k-test-{source_index}",
        "source_index": source_index,
        "question": row["question"],
        "answer": row["answer"],
    }


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    """Write records as UTF-8 JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_gsm8k_split(dataset_name: str, dataset_config: str, split: str):
    """Load a GSM8K split via Hugging Face datasets."""

    from datasets import load_dataset

    return load_dataset(dataset_name, dataset_config, split=split)


def build_eval_records(dataset: Sequence[dict], sample_size: int, seed: int) -> list[dict]:
    """Sample and normalize records from a loaded dataset split."""

    indices = sample_indices(len(dataset), sample_size, seed)
    return [normalize_record(dataset[i], i) for i in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the JSONL eval subset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_gsm8k_split(args.dataset_name, args.dataset_config, args.split)
    records = build_eval_records(dataset, args.sample_size, args.seed)
    write_jsonl(records, args.output)
    print(
        f"Wrote {len(records)} records from {args.dataset_name}/{args.dataset_config} "
        f"{args.split} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
