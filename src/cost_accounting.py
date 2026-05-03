"""Cost accounting helpers for compute-matched experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PRICES_PATH = Path("prices.yaml")
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class TrainingCost:
    """One-time cost for generating data and fine-tuning."""

    data_generation_usd: float
    gpu_usd: float

    @property
    def total_usd(self) -> float:
        return self.data_generation_usd + self.gpu_usd


@dataclass(frozen=True)
class InferenceCost:
    """Per-query cost for self-hosted sampling plus optional LLM judge calls."""

    model_usd: float
    judge_usd: float

    @property
    def total_usd(self) -> float:
        return self.model_usd + self.judge_usd


def load_prices(path: Path | str = DEFAULT_PRICES_PATH) -> dict[str, Any]:
    """Load pricing assumptions from YAML."""

    with Path(path).open("r", encoding="utf-8") as f:
        prices = yaml.safe_load(f)
    if not isinstance(prices, dict):
        raise ValueError(f"Price config must be a mapping: {path}")
    return prices


def openai_token_cost_usd(
    prices: dict[str, Any],
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Compute OpenAI API token cost."""

    model_prices = prices["openai"][model]
    input_cost = input_tokens / TOKENS_PER_MILLION * model_prices["input_per_1m_usd"]
    output_cost = output_tokens / TOKENS_PER_MILLION * model_prices["output_per_1m_usd"]
    return input_cost + output_cost


def self_hosted_token_cost_usd(
    prices: dict[str, Any],
    model_key: str,
    tokens: int,
) -> float:
    """Compute self-hosted model serving cost from derived per-token price."""

    per_1m = prices["self_hosted"][model_key]["derived_per_1m_tokens_usd"]
    return tokens / TOKENS_PER_MILLION * per_1m


def training_cost_usd(
    prices: dict[str, Any],
    synthetic_examples: int,
    gpu_hours: float,
    self_hosted_model_key: str = "qwen_2_5_1_5b",
) -> TrainingCost:
    """Compute one-time train cost for one fine-tuned model."""

    if synthetic_examples < 0:
        raise ValueError("synthetic_examples cannot be negative")
    if gpu_hours < 0:
        raise ValueError("gpu_hours cannot be negative")

    data_generation = synthetic_examples * prices["data_generation"]["per_example_usd"]
    gpu = gpu_hours * prices["self_hosted"][self_hosted_model_key]["gpu_hourly_usd"]
    return TrainingCost(data_generation_usd=data_generation, gpu_usd=gpu)


def inference_cost_per_query_usd(
    prices: dict[str, Any],
    budget: int,
    model_tokens_per_sample: int,
    self_hosted_model_key: str = "qwen_2_5_1_5b",
    judge_model: str | None = None,
    judge_input_tokens: int = 0,
    judge_output_tokens: int = 0,
) -> InferenceCost:
    """Compute per-query inference cost for a single strategy.

    `budget` is the number of samples drawn from the self-hosted model. For BoN,
    pass `judge_model` and judge token counts to include one judge call.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    if model_tokens_per_sample < 0:
        raise ValueError("model_tokens_per_sample cannot be negative")
    if judge_input_tokens < 0 or judge_output_tokens < 0:
        raise ValueError("judge token counts cannot be negative")

    model_tokens = budget * model_tokens_per_sample
    model_cost = self_hosted_token_cost_usd(
        prices,
        model_key=self_hosted_model_key,
        tokens=model_tokens,
    )

    judge_cost = 0.0
    if judge_model is not None:
        judge_cost = openai_token_cost_usd(
            prices,
            model=judge_model,
            input_tokens=judge_input_tokens,
            output_tokens=judge_output_tokens,
        )

    return InferenceCost(model_usd=model_cost, judge_usd=judge_cost)


def total_serving_cost_usd(
    train_cost_usd: float,
    inference_cost_per_query_usd: float,
    query_count: int,
) -> float:
    """Compute total cost for serving `query_count` queries."""

    if train_cost_usd < 0:
        raise ValueError("train_cost_usd cannot be negative")
    if inference_cost_per_query_usd < 0:
        raise ValueError("inference_cost_per_query_usd cannot be negative")
    if query_count < 0:
        raise ValueError("query_count cannot be negative")

    return train_cost_usd + query_count * inference_cost_per_query_usd
