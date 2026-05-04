import tempfile
import unittest
from pathlib import Path

from src.train_lora import (
    LoRATrainingConfig,
    build_lora_kwargs,
    prepare_lora_training,
    validate_training_file,
)


def write_jsonl(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class TrainLoRATests(unittest.TestCase):
    def test_validate_training_file_accepts_chat_template_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "train.jsonl"
            write_jsonl(
                data_path,
                [
                    '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}'
                ],
            )

            self.assertEqual(validate_training_file(data_path), 1)

    def test_validate_training_file_rejects_missing_assistant(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "train.jsonl"
            write_jsonl(data_path, ['{"messages":[{"role":"user","content":"q"}]}'])

            with self.assertRaises(ValueError):
                validate_training_file(data_path)

    def test_build_lora_kwargs_sets_training_hub_chat_template_fields(self):
        config = LoRATrainingConfig(
            data_path=Path("data/train.jsonl"),
            ckpt_output_dir=Path("checkpoints/n3"),
        )

        kwargs = build_lora_kwargs(config)

        self.assertEqual(kwargs["model_path"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(kwargs["data_path"], "data/train.jsonl")
        self.assertEqual(kwargs["ckpt_output_dir"], "checkpoints/n3")
        self.assertEqual(kwargs["lora_r"], 16)
        self.assertEqual(kwargs["lora_alpha"], 32)
        self.assertEqual(kwargs["num_epochs"], 3)
        self.assertEqual(kwargs["learning_rate"], 2e-4)
        self.assertEqual(kwargs["dataset_type"], "chat_template")
        self.assertEqual(kwargs["field_messages"], "messages")
        self.assertTrue(kwargs["load_in_4bit"])
        self.assertFalse(kwargs["bf16"])
        self.assertTrue(kwargs["fp16"])

    def test_prepare_lora_training_is_dry_run_plan(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "train.jsonl"
            write_jsonl(
                data_path,
                [
                    '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}'
                ],
            )
            config = LoRATrainingConfig(data_path=data_path)

            plan = prepare_lora_training(config)

        self.assertEqual(plan.record_count, 1)
        self.assertEqual(plan.lora_kwargs["data_path"], str(data_path))


if __name__ == "__main__":
    unittest.main()
