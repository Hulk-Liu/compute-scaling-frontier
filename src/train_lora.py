"""Thin wrapper around training_hub.lora_sft for Colab LoRA training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path("checkpoints/qwen2.5-1.5b-gsm8k-lora")


@dataclass(frozen=True)
class LoRATrainingConfig:
    """Serializable training configuration for one LoRA run."""

    data_path: Path
    ckpt_output_dir: Path = DEFAULT_OUTPUT_DIR
    model_path: str = DEFAULT_MODEL_PATH
    backend: str = "unsloth"
    lora_r: int = 16
    lora_alpha: int = 32
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_len: int = 2048
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    load_in_4bit: bool = True
    bf16: bool = False
    fp16: bool = True
    dataset_type: str = "chat_template"
    field_messages: str = "messages"


@dataclass(frozen=True)
class LoRATrainingPlan:
    """Dry-run training plan."""

    record_count: int
    lora_kwargs: dict[str, Any]


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL records."""

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


def validate_chat_template_record(record: dict[str, Any], row_index: int) -> None:
    """Validate the subset of chat-template schema training_hub needs."""

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row {row_index} is missing non-empty list field 'messages'")

    has_assistant = False
    for message_index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"Row {row_index} message {message_index} is not an object")

        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"Row {row_index} message {message_index} has unsupported role {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Row {row_index} message {message_index} is missing string content"
            )
        has_assistant = has_assistant or role == "assistant"

    if not has_assistant:
        raise ValueError(f"Row {row_index} must include an assistant message")


def validate_training_file(path: Path | str) -> int:
    """Validate generated training JSONL and return record count."""

    records = read_jsonl(path)
    if not records:
        raise ValueError(f"Training file is empty: {path}")
    for index, record in enumerate(records, start=1):
        validate_chat_template_record(record, row_index=index)
    return len(records)


def build_lora_kwargs(config: LoRATrainingConfig) -> dict[str, Any]:
    """Build kwargs for training_hub.lora_sft."""

    return {
        "model_path": config.model_path,
        "data_path": str(config.data_path),
        "ckpt_output_dir": str(config.ckpt_output_dir),
        "backend": config.backend,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "max_seq_len": config.max_seq_len,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "load_in_4bit": config.load_in_4bit,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "dataset_type": config.dataset_type,
        "field_messages": config.field_messages,
    }


def prepare_lora_training(config: LoRATrainingConfig) -> LoRATrainingPlan:
    """Validate data and prepare the lora_sft call without executing training."""

    record_count = validate_training_file(config.data_path)
    return LoRATrainingPlan(
        record_count=record_count,
        lora_kwargs=build_lora_kwargs(config),
    )


def run_lora_training(config: LoRATrainingConfig, execute: bool = False) -> LoRATrainingPlan:
    """Prepare, and optionally execute, a LoRA training run."""

    plan = prepare_lora_training(config)
    if not execute:
        return plan

    if config.backend == "unsloth":
        import unsloth  # noqa: F401

    from training_hub import lora_sft

    lora_sft(**plan.lora_kwargs)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ckpt-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--backend", default="unsloth")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 precision. Requires Ampere+ GPU; default is fp16 for T4.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 precision. Defaults to fp16 for T4 compatibility.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call training_hub.lora_sft. Defaults to dry-run only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = LoRATrainingConfig(
        data_path=args.data_path,
        ckpt_output_dir=args.ckpt_output_dir,
        model_path=args.model_path,
        backend=args.backend,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_in_4bit=not args.no_4bit,
        bf16=args.bf16,
        fp16=not args.no_fp16,
    )
    plan = run_lora_training(config, execute=args.execute)
    mode = "execute" if args.execute else "dry-run"
    print(f"Prepared {mode} LoRA run for {plan.record_count} records")
    print(json.dumps(plan.lora_kwargs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
