import unittest
from pathlib import Path

from src.smoke_lora_adapter import build_raw_row


class SmokeLoRAAdapterTests(unittest.TestCase):
    def test_build_raw_row_uses_shared_inference_schema(self):
        row = build_raw_row(
            eval_row={
                "id": "gsm8k-test-1",
                "source_index": 1,
                "question": "q",
                "answer": "#### 12",
            },
            prediction="Reasoning\n#### 12",
            adapter_path=Path("checkpoints/lora"),
            max_new_tokens=512,
        )

        self.assertEqual(row["model"], "checkpoints/lora")
        self.assertEqual(row["strategy"], "lora_greedy")
        self.assertEqual(row["budget"], 1)
        self.assertTrue(row["is_correct"])
        self.assertTrue(row["answer_format_ok"])
        self.assertEqual(row["raw_response"]["adapter_path"], "checkpoints/lora")
        self.assertEqual(row["raw_response"]["max_new_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
