import csv
import tempfile
import unittest
from pathlib import Path

from src.plot_results import (
    build_metric_matrix,
    read_aggregate_rows,
    strategy_label,
)


class PlotResultsTests(unittest.TestCase):
    def test_strategy_label_formats_budgeted_sc(self):
        self.assertEqual(strategy_label({"strategy": "greedy", "budget": 1}), "Greedy")
        self.assertEqual(strategy_label({"strategy": "sc", "budget": 8}), "SC@8")

    def test_build_metric_matrix_orders_rows_and_columns(self):
        rows = [
            {"train_size": 0, "strategy": "greedy", "budget": 1, "accuracy": 0.5},
            {"train_size": 0, "strategy": "sc", "budget": 4, "accuracy": 0.6},
            {"train_size": 100, "strategy": "greedy", "budget": 1, "accuracy": 0.7},
            {"train_size": 100, "strategy": "sc", "budget": 4, "accuracy": 0.8},
        ]

        matrix = build_metric_matrix(
            rows,
            metric="accuracy",
            train_sizes=[0, 100],
            strategy_labels=["Greedy", "SC@4"],
        )

        self.assertEqual(matrix, [[0.5, 0.6], [0.7, 0.8]])

    def test_read_aggregate_rows_coerces_numeric_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "aggregated.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "train_size",
                        "strategy",
                        "budget",
                        "accuracy",
                        "answer_format_ok_rate",
                        "train_cost_usd",
                        "inference_cost_per_query_usd",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "train_size": "500",
                        "strategy": "sc",
                        "budget": "8",
                        "accuracy": "0.7",
                        "answer_format_ok_rate": "1.0",
                        "train_cost_usd": "0.55",
                        "inference_cost_per_query_usd": "0.0",
                    }
                )

            rows = read_aggregate_rows(path)

        self.assertEqual(rows[0]["train_size"], 500)
        self.assertEqual(rows[0]["budget"], 8)
        self.assertEqual(rows[0]["accuracy"], 0.7)


if __name__ == "__main__":
    unittest.main()
