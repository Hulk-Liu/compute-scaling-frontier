import tempfile
import unittest
from pathlib import Path

from src.filter_training_data import (
    default_output_paths,
    filter_training_file,
    split_valid_invalid,
)
from src.prepare_eval_set import write_jsonl
from src.validate_training_data import read_training_records


class FilterTrainingDataTests(unittest.TestCase):
    def test_default_output_paths(self):
        valid, invalid = default_output_paths(Path("data/augmented_train_100.jsonl"))

        self.assertEqual(valid, Path("data/augmented_train_100_valid.jsonl"))
        self.assertEqual(invalid, Path("data/augmented_train_100_invalid.jsonl"))

    def test_split_valid_invalid(self):
        valid, invalid, summary = split_valid_invalid(
            [
                {"teacher_response": "Reasoning\n#### 12", "gold_answer": "#### 12"},
                {"teacher_response": "Reasoning\n#### 9", "gold_answer": "#### 10"},
            ]
        )

        self.assertEqual(len(valid), 1)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.valid, 1)
        self.assertEqual(summary.invalid, 1)
        self.assertEqual(summary.accuracy, 0.5)
        self.assertEqual(invalid[0]["validation"]["row_index"], 2)
        self.assertEqual(invalid[0]["validation"]["extracted_prediction"], "9")
        self.assertEqual(invalid[0]["validation"]["extracted_gold"], "10")

    def test_filter_training_file_writes_valid_and_invalid_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "train.jsonl"
            valid_path = Path(tmp_dir) / "train_valid.jsonl"
            invalid_path = Path(tmp_dir) / "train_invalid.jsonl"
            write_jsonl(
                [
                    {"teacher_response": "Reasoning\n#### 12", "gold_answer": "#### 12"},
                    {"teacher_response": "Reasoning\n#### 9", "gold_answer": "#### 10"},
                ],
                input_path,
            )

            summary = filter_training_file(
                input_path,
                valid_output=valid_path,
                invalid_output=invalid_path,
            )

            valid_records = read_training_records(valid_path)
            invalid_records = read_training_records(invalid_path)

        self.assertEqual(summary.valid, 1)
        self.assertEqual(summary.invalid, 1)
        self.assertEqual(valid_records[0]["teacher_response"], "Reasoning\n#### 12")
        self.assertEqual(invalid_records[0]["validation"]["row_index"], 2)


if __name__ == "__main__":
    unittest.main()
