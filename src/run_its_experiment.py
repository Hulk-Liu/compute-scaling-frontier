"""Run a tiny inference/evaluation experiment and write raw JSONL outputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.evaluator import (
    build_answer_diagnostics,
    extract_last_number,
)
from src.prepare_eval_set import DEFAULT_OUTPUT_PATH as DEFAULT_EVAL_PATH


DEFAULT_OUTPUT_PATH = Path("results/raw/_smoke_gpt4omini_greedy.jsonl")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_MAX_TOKENS = 512
DEFAULT_BUDGET = 1


def read_eval_records(path: Path | str, n_eval: int | None = None) -> list[dict[str, Any]]:
    """Read eval records from JSONL."""

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
            if n_eval is not None and len(rows) >= n_eval:
                break
    if not rows:
        raise ValueError(f"No eval records found in {path}")
    return rows


def build_math_prompt(question: str) -> str:
    """Build the GSM8K inference prompt."""

    return (
        "Solve this grade-school math problem. "
        "Show concise reasoning and end with a final line in the exact format "
        "'#### <number>'.\n\n"
        f"Problem:\n{question}"
    )


def _response_content(response: dict[str, Any]) -> str:
    content = response.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(f"Expected string response content, got {type(content).__name__}")
    return content


def final_answer_projection(response: str) -> str:
    """Project free-form math output into the final-answer space for SC voting."""

    return extract_last_number(response) or response.strip()


def _base_raw_row(
    eval_row: dict[str, Any],
    prediction: str,
    model: str,
    strategy: str,
    budget: int,
    raw_response: dict[str, Any],
) -> dict[str, Any]:
    """Build the common raw result row schema."""

    return {
        "id": eval_row["id"],
        "source_index": eval_row["source_index"],
        "question": eval_row["question"],
        "answer": eval_row["answer"],
        "prediction": prediction,
        "model": model,
        "strategy": strategy,
        "budget": budget,
        "raw_response": raw_response,
        **build_answer_diagnostics(prediction, eval_row["answer"]),
    }


async def run_greedy_openai_compatible(
    eval_rows: list[dict[str, Any]],
    model: str,
    endpoint: str,
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Run greedy inference through its_hub's OpenAI-compatible LM wrapper."""

    from its_hub import OpenAICompatibleLanguageModel
    from its_hub.api.types import ChatMessage

    lm = OpenAICompatibleLanguageModel(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model,
        max_tokens=max_tokens,
        temperature=0.0,
        max_concurrency=1,
    )

    raw_rows = []
    try:
        for eval_row in eval_rows:
            prompt = build_math_prompt(eval_row["question"])
            response = await lm.agenerate_single(
                [ChatMessage(role="user", content=prompt)]
            )
            raw_rows.append(
                _base_raw_row(
                    eval_row=eval_row,
                    prediction=_response_content(response),
                    model=model,
                    strategy="greedy",
                    budget=1,
                    raw_response=response,
                )
            )
    finally:
        await lm.close()

    return raw_rows


async def run_self_consistency_openai_compatible(
    eval_rows: list[dict[str, Any]],
    model: str,
    endpoint: str,
    api_key: str,
    budget: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Run Self-Consistency through its_hub."""

    if budget <= 1:
        raise ValueError("Self-Consistency budget must be greater than 1")

    from its_hub import OpenAICompatibleLanguageModel, SelfConsistency

    lm = OpenAICompatibleLanguageModel(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model,
        max_tokens=max_tokens,
        temperature=0.7,
        max_concurrency=budget,
    )
    algorithm = SelfConsistency(
        consistency_space_projection_func=final_answer_projection
    )

    raw_rows = []
    try:
        for eval_row in eval_rows:
            prompt = build_math_prompt(eval_row["question"])
            result = await algorithm.ainfer(
                lm,
                prompt,
                budget=budget,
                return_response_only=False,
            )
            selected = result.the_one
            raw_rows.append(
                {
                    **_base_raw_row(
                        eval_row=eval_row,
                        prediction=_response_content(selected),
                        model=model,
                        strategy="sc",
                        budget=budget,
                        raw_response=selected,
                    ),
                    "all_responses": result.responses,
                    "response_counts": {
                        str(key): value for key, value in result.response_counts.items()
                    },
                    "selected_index": result.selected_index,
                }
            )
    finally:
        await lm.close()

    return raw_rows


def write_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> None:
    """Write raw result records as JSONL."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--n-eval", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--strategy", choices=["greedy", "sc"], default="greedy")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for this smoke inference step")

    eval_rows = read_eval_records(args.eval_path, n_eval=args.n_eval)
    if args.strategy == "greedy":
        raw_rows = asyncio.run(
            run_greedy_openai_compatible(
                eval_rows=eval_rows,
                model=args.model,
                endpoint=args.endpoint,
                api_key=api_key,
                max_tokens=args.max_tokens,
            )
        )
    else:
        raw_rows = asyncio.run(
            run_self_consistency_openai_compatible(
                eval_rows=eval_rows,
                model=args.model,
                endpoint=args.endpoint,
                api_key=api_key,
                budget=args.budget,
                max_tokens=args.max_tokens,
            )
        )
    write_jsonl(raw_rows, args.output)
    print(f"Wrote {len(raw_rows)} raw inference rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
