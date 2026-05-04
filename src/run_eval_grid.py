"""Run the required Qwen eval grid through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.aggregate_results import aggregate_cell, read_jsonl, write_aggregated_csv
from src.cost_accounting import DEFAULT_PRICES_PATH, load_prices
from src.prepare_eval_set import DEFAULT_OUTPUT_PATH as DEFAULT_EVAL_PATH
from src.run_its_experiment import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_JUDGE_ENDPOINT,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_MODEL,
    read_eval_records,
    run_best_of_n_openai_compatible,
    run_greedy_openai_compatible,
    run_self_consistency_openai_compatible,
    write_jsonl,
)


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_RAW_DIR = Path("results/raw")
DEFAULT_AGGREGATED_PATH = Path("results/aggregated.csv")


@dataclass(frozen=True)
class ModelVariant:
    """One model variant in the compute-matched eval grid."""

    name: str
    model_id: str
    train_size: int
    train_gpu_hours: float


@dataclass(frozen=True)
class StrategySpec:
    """One inference-time strategy in the eval grid."""

    name: str
    budget: int
    judge_model: str | None = None


def default_model_variants(
    base_model: str,
    lora_n100_model: str,
    lora_n500_model: str,
    train_gpu_hours_n100: float,
    train_gpu_hours_n500: float,
) -> list[ModelVariant]:
    """Build the default base/n100/n500 model variant list."""

    return [
        ModelVariant(
            name="base",
            model_id=base_model,
            train_size=0,
            train_gpu_hours=0.0,
        ),
        ModelVariant(
            name="lora_n100",
            model_id=lora_n100_model,
            train_size=100,
            train_gpu_hours=train_gpu_hours_n100,
        ),
        ModelVariant(
            name="lora_n500",
            model_id=lora_n500_model,
            train_size=500,
            train_gpu_hours=train_gpu_hours_n500,
        ),
    ]


def default_strategy_specs(
    include_bon4: bool = False,
    bon_judge_model: str = DEFAULT_JUDGE_MODEL,
) -> list[StrategySpec]:
    """Build the required inference strategy list."""

    strategies = [
        StrategySpec(name="greedy", budget=1),
        StrategySpec(name="sc", budget=4),
        StrategySpec(name="sc", budget=8),
    ]
    if include_bon4:
        strategies.append(
            StrategySpec(name="bon", budget=4, judge_model=bon_judge_model)
        )
    return strategies


def cell_slug(model: ModelVariant, strategy: StrategySpec) -> str:
    """Build a stable file slug for one model/strategy cell."""

    if strategy.name == "greedy":
        strategy_slug = "greedy"
    else:
        strategy_slug = f"{strategy.name}{strategy.budget}"
    return f"qwen_{model.name}_{strategy_slug}"


async def run_grid_cell(
    eval_rows: list[dict[str, Any]],
    model: ModelVariant,
    strategy: StrategySpec,
    endpoint: str,
    api_key: str,
    max_tokens: int,
    judge_endpoint: str,
    judge_api_key: str,
    judge_max_tokens: int,
) -> list[dict[str, Any]]:
    """Run one model/strategy cell and return raw result rows."""

    if strategy.name == "greedy":
        return await run_greedy_openai_compatible(
            eval_rows=eval_rows,
            model=model.model_id,
            endpoint=endpoint,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    if strategy.name == "sc":
        return await run_self_consistency_openai_compatible(
            eval_rows=eval_rows,
            model=model.model_id,
            endpoint=endpoint,
            api_key=api_key,
            budget=strategy.budget,
            max_tokens=max_tokens,
        )
    if strategy.name == "bon":
        if strategy.judge_model is None:
            raise ValueError("Best-of-N strategy requires a judge model")
        return await run_best_of_n_openai_compatible(
            eval_rows=eval_rows,
            model=model.model_id,
            endpoint=endpoint,
            api_key=api_key,
            budget=strategy.budget,
            judge_model=strategy.judge_model,
            judge_endpoint=judge_endpoint,
            judge_api_key=judge_api_key,
            max_tokens=max_tokens,
            judge_max_tokens=judge_max_tokens,
        )
    raise ValueError(f"Unsupported strategy: {strategy.name}")


async def run_eval_grid(
    eval_rows: list[dict[str, Any]],
    models: list[ModelVariant],
    strategies: list[StrategySpec],
    endpoint: str,
    api_key: str,
    raw_dir: Path,
    prices: dict[str, Any],
    output_csv: Path,
    model_tokens_per_sample: int,
    max_tokens: int,
    judge_endpoint: str,
    judge_api_key: str,
    judge_max_tokens: int,
    resume: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Run all cells and write one aggregate CSV."""

    aggregate_rows = []
    for model in models:
        for strategy in strategies:
            slug = cell_slug(model, strategy)
            raw_path = raw_dir / f"{slug}.jsonl"
            print(
                f"[grid] model={model.model_id} train_size={model.train_size} "
                f"strategy={strategy.name} budget={strategy.budget} raw={raw_path}"
            )

            if dry_run:
                continue

            if resume and raw_path.exists():
                print(f"[grid] resume: using existing {raw_path}")
                raw_rows = read_jsonl(raw_path)
            else:
                raw_rows = await run_grid_cell(
                    eval_rows=eval_rows,
                    model=model,
                    strategy=strategy,
                    endpoint=endpoint,
                    api_key=api_key,
                    max_tokens=max_tokens,
                    judge_endpoint=judge_endpoint,
                    judge_api_key=judge_api_key,
                    judge_max_tokens=judge_max_tokens,
                )
                write_jsonl(raw_rows, raw_path)
                print(f"[grid] wrote {len(raw_rows)} raw rows to {raw_path}")

            aggregate_rows.append(
                aggregate_cell(
                    raw_rows=raw_rows,
                    prices=prices,
                    train_size=model.train_size,
                    strategy=strategy.name,
                    budget=strategy.budget,
                    train_gpu_hours=model.train_gpu_hours,
                    model_tokens_per_sample=model_tokens_per_sample,
                    judge_model=strategy.judge_model,
                )
            )

    if not dry_run:
        write_aggregated_csv(aggregate_rows, output_csv)
        print(f"[grid] wrote {len(aggregate_rows)} aggregate rows to {output_csv}")
    return aggregate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora-n100-model", default="qwen-gsm8k-lora-n100")
    parser.add_argument("--lora-n500-model", default="qwen-gsm8k-lora-n500")
    parser.add_argument("--train-gpu-hours-n100", type=float, default=0.03)
    parser.add_argument("--train-gpu-hours-n500", type=float, default=0.125)
    parser.add_argument("--model-tokens-per-sample", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--include-bon4",
        action="store_true",
        help="Add optional judge-assisted Best-of-N @4 cells to the grid.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-endpoint", default=DEFAULT_JUDGE_ENDPOINT)
    parser.add_argument("--judge-api-key-env", default="JUDGE_OPENAI_API_KEY")
    parser.add_argument("--judge-max-tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_AGGREGATED_PATH)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing raw JSONL cells instead of overwriting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grid without calling the endpoint.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise RuntimeError(f"{args.api_key_env} must be set for eval grid runs")
    judge_api_key = os.environ.get(args.judge_api_key_env)
    if args.include_bon4 and not judge_api_key and not args.dry_run:
        raise RuntimeError(
            f"{args.judge_api_key_env} must be set for Best-of-N judge calls"
        )

    eval_rows = read_eval_records(args.eval_path, n_eval=args.n_eval)
    models = default_model_variants(
        base_model=args.base_model,
        lora_n100_model=args.lora_n100_model,
        lora_n500_model=args.lora_n500_model,
        train_gpu_hours_n100=args.train_gpu_hours_n100,
        train_gpu_hours_n500=args.train_gpu_hours_n500,
    )
    strategies = default_strategy_specs(
        include_bon4=args.include_bon4,
        bon_judge_model=args.judge_model,
    )
    prices = load_prices(args.prices)

    asyncio.run(
        run_eval_grid(
            eval_rows=eval_rows,
            models=models,
            strategies=strategies,
            endpoint=args.endpoint,
            api_key=api_key or "",
            raw_dir=args.raw_dir,
            prices=prices,
            output_csv=args.output,
            model_tokens_per_sample=args.model_tokens_per_sample,
            max_tokens=args.max_tokens,
            judge_endpoint=args.judge_endpoint,
            judge_api_key=judge_api_key or "",
            judge_max_tokens=args.judge_max_tokens,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
