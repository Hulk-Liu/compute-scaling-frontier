"""Run a tiny greedy smoke test against a local Unsloth LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.prepare_eval_set import DEFAULT_OUTPUT_PATH as DEFAULT_EVAL_PATH
from src.run_its_experiment import (
    _base_raw_row,
    build_math_prompt,
    read_eval_records,
    write_jsonl,
)


DEFAULT_OUTPUT_PATH = Path("results/raw/_smoke_qwen_lora_greedy.jsonl")
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_SEQ_LENGTH = 2048


def decode_generated_text(tokenizer: Any, outputs: Any, input_length: int) -> str:
    """Decode only the newly generated tokens."""

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_raw_row(
    eval_row: dict[str, Any],
    prediction: str,
    adapter_path: Path | str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build the raw row schema for local LoRA greedy inference."""

    return _base_raw_row(
        eval_row=eval_row,
        prediction=prediction,
        model=str(adapter_path),
        strategy="lora_greedy",
        budget=1,
        raw_response={
            "content": prediction,
            "adapter_path": str(adapter_path),
            "max_new_tokens": max_new_tokens,
        },
    )


def run_lora_adapter_smoke(
    adapter_path: Path | str,
    eval_rows: list[dict[str, Any]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
) -> list[dict[str, Any]]:
    """Load a local LoRA adapter with Unsloth and run greedy generation."""

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    raw_rows = []
    for eval_row in eval_rows:
        prompt = build_math_prompt(eval_row["question"])
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([rendered_prompt], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        prediction = decode_generated_text(
            tokenizer=tokenizer,
            outputs=outputs,
            input_length=inputs.input_ids.shape[-1],
        )
        raw_rows.append(
            build_raw_row(
                eval_row=eval_row,
                prediction=prediction,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
            )
        )

    return raw_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--n-eval", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_rows = read_eval_records(args.eval_path, n_eval=args.n_eval)
    raw_rows = run_lora_adapter_smoke(
        adapter_path=args.adapter_path,
        eval_rows=eval_rows,
        max_new_tokens=args.max_new_tokens,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )
    write_jsonl(raw_rows, args.output)
    print(f"Wrote {len(raw_rows)} local LoRA smoke rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
