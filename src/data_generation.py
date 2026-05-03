"""Generate small GSM8K math-reasoning SFT data with sdg_hub."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.prepare_eval_set import sample_indices, write_jsonl


DEFAULT_FLOW_PATH = Path("src/flows/math_reasoning_distill.yaml")
DEFAULT_OUTPUT_PATH = Path("data/_smoke_augmented_train.jsonl")
DEFAULT_DATASET_NAME = "gsm8k"
DEFAULT_DATASET_CONFIG = "main"
DEFAULT_SPLIT = "train"
DEFAULT_SEED = 20260503
DEFAULT_N = 3
DEFAULT_MODEL = "openai/gpt-4o-mini"


def load_gsm8k_split(dataset_name: str, dataset_config: str, split: str):
    """Load a GSM8K split via Hugging Face datasets."""

    from datasets import load_dataset

    return load_dataset(dataset_name, dataset_config, split=split)


def sample_gsm8k_rows(dataset: Sequence[dict], n: int, seed: int) -> pd.DataFrame:
    """Sample GSM8K rows into the dataframe shape expected by the sdg_hub flow."""

    indices = sample_indices(len(dataset), n, seed)
    rows = []
    for index in indices:
        row = dataset[index]
        rows.append(
            {
                "source_index": index,
                "question": row["question"],
                "answer": row["answer"],
            }
        )
    return pd.DataFrame(rows)


def run_flow(
    input_df: pd.DataFrame,
    flow_path: Path,
    model: str,
    max_concurrency: int = 1,
) -> pd.DataFrame:
    """Run the configured sdg_hub flow."""

    from sdg_hub import Flow

    flow = Flow.from_yaml(str(flow_path))
    flow.set_model_config(model=model)
    return flow.generate(input_df, max_concurrency=max_concurrency)


def to_training_records(flow_output: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert sdg_hub output rows to training_hub chat-template records."""

    records = []
    for row in flow_output.to_dict("records"):
        teacher_response = str(row["teacher_response_content"]).strip()
        records.append(
            {
                "id": f"gsm8k-train-{row['source_index']}",
                "source_index": row["source_index"],
                "question": row["question"],
                "gold_answer": row["answer"],
                "teacher_response": teacher_response,
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": teacher_response},
                ],
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-concurrency", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_gsm8k_split(args.dataset_name, args.dataset_config, args.split)
    input_df = sample_gsm8k_rows(dataset, n=args.n, seed=args.seed)
    flow_output = run_flow(
        input_df=input_df,
        flow_path=args.flow,
        model=args.model,
        max_concurrency=args.max_concurrency,
    )
    records = to_training_records(flow_output)
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} training records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
