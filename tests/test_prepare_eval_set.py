import tempfile
import unittest
from pathlib import Path

from src.prepare_eval_set import build_eval_records, sample_indices, write_jsonl


class PrepareEvalSetTests(unittest.TestCase):
    def test_sample_indices_is_deterministic_and_sorted(self):
        first = sample_indices(population_size=100, sample_size=5, seed=123)
        second = sample_indices(population_size=100, sample_size=5, seed=123)

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)

    def test_sample_indices_rejects_invalid_sample_size(self):
        with self.assertRaises(ValueError):
            sample_indices(population_size=3, sample_size=4, seed=123)

    def test_build_eval_records_keeps_expected_schema(self):
        dataset = [
            {"question": "q0", "answer": "#### 0", "extra": "unused"},
            {"question": "q1", "answer": "#### 1", "extra": "unused"},
            {"question": "q2", "answer": "#### 2", "extra": "unused"},
        ]

        records = build_eval_records(dataset, sample_size=2, seed=7)

        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(set(record), {"id", "source_index", "question", "answer"})
            self.assertTrue(record["id"].startswith("gsm8k-test-"))

    def test_write_jsonl_writes_one_line_per_record(self):
        records = [
            {"id": "a", "source_index": 0, "question": "q0", "answer": "#### 0"},
            {"id": "b", "source_index": 1, "question": "q1", "answer": "#### 1"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "eval.jsonl"
            write_jsonl(records, output)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertIn('"id": "a"', lines[0])


if __name__ == "__main__":
    unittest.main()
