import tempfile
import unittest
from pathlib import Path

from src.aggregate_results import aggregate_cell, read_jsonl, write_aggregated_csv


PRICES = {
    "openai": {
        "gpt-4o-mini": {
            "input_per_1m_usd": 0.15,
            "output_per_1m_usd": 0.60,
        }
    },
    "self_hosted": {
        "qwen_2_5_1_5b": {
            "gpu_hourly_usd": 0.40,
            "derived_per_1m_tokens_usd": 0.18,
        }
    },
    "data_generation": {
        "per_example_usd": 0.001,
    },
}


class AggregateResultsTests(unittest.TestCase):
    def test_read_jsonl_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.jsonl"
            path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

            with self.assertRaises(ValueError):
                read_jsonl(path)

    def test_aggregate_cell_evaluates_accuracy_and_costs(self):
        rows = [
            {"id": "gsm8k-test-1", "prediction": "Answer: 12", "gold_answer": "#### 12"},
            {"id": "gsm8k-test-2", "prediction": "Answer: 9", "gold_answer": "#### 10"},
        ]

        row = aggregate_cell(
            raw_rows=rows,
            prices=PRICES,
            train_size=100,
            strategy="sc",
            budget=4,
            train_gpu_hours=0.5,
            model_tokens_per_sample=250,
            query_counts=[1_000, 10_000],
        )

        self.assertEqual(row["train_size"], 100)
        self.assertEqual(row["strategy"], "sc")
        self.assertEqual(row["budget"], 4)
        self.assertEqual(row["n_eval"], 2)
        self.assertEqual(row["correct"], 1)
        self.assertEqual(row["accuracy"], 0.5)
        self.assertAlmostEqual(row["train_cost_usd"], 0.3)
        self.assertAlmostEqual(row["inference_cost_per_query_usd"], 0.00018)
        self.assertAlmostEqual(row["total_cost_usd_at_1000"], 0.48)
        self.assertAlmostEqual(row["total_cost_usd_at_10000"], 2.1)

    def test_aggregate_cell_accepts_answer_as_gold_alias(self):
        row = aggregate_cell(
            raw_rows=[{"id": "1", "prediction": "12", "answer": "#### 12"}],
            prices=PRICES,
            train_size=0,
            strategy="greedy",
            budget=1,
            train_gpu_hours=0.0,
            model_tokens_per_sample=100,
            query_counts=[1_000],
        )

        self.assertEqual(row["correct"], 1)

    def test_aggregate_cell_requires_prediction(self):
        with self.assertRaises(ValueError):
            aggregate_cell(
                raw_rows=[{"id": "1", "answer": "#### 12"}],
                prices=PRICES,
                train_size=0,
                strategy="greedy",
                budget=1,
                train_gpu_hours=0.0,
                model_tokens_per_sample=100,
            )

    def test_write_aggregated_csv(self):
        rows = [{"train_size": 0, "strategy": "greedy", "accuracy": 0.5}]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "aggregated.csv"
            write_aggregated_csv(rows, output)
            content = output.read_text(encoding="utf-8")

        self.assertIn("train_size,strategy,accuracy", content)
        self.assertIn("0,greedy,0.5", content)


if __name__ == "__main__":
    unittest.main()
