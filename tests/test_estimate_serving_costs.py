import unittest
from pathlib import Path
from typing import Any

from src.estimate_serving_costs import (
    CharacterHeuristicCounter,
    candidate_responses,
    estimate_bon_judge_tokens_per_query,
    raw_path_for_row,
    update_row_costs,
)


class FixedTokenCounter:
    method_name = "fixed_test_counter"

    def count(self, text: str) -> int:
        return 10 if text else 0


class EstimateServingCostsTests(unittest.TestCase):
    def test_character_counter_rounds_up_and_keeps_nonempty_text_nonzero(self):
        counter = CharacterHeuristicCounter(chars_per_token=4.0)

        self.assertEqual(counter.count(""), 0)
        self.assertEqual(counter.count("a"), 1)
        self.assertEqual(counter.count("abcd"), 1)
        self.assertEqual(counter.count("abcde"), 2)

    def test_raw_path_for_row_matches_grid_outputs(self):
        raw_dir = Path("results/raw")

        self.assertEqual(
            raw_path_for_row(
                {"train_size": "0", "strategy": "greedy", "budget": "1"},
                raw_dir,
            ),
            raw_dir / "qwen_base_greedy.jsonl",
        )
        self.assertEqual(
            raw_path_for_row(
                {"train_size": "500", "strategy": "sc", "budget": "8"},
                raw_dir,
            ),
            raw_dir / "qwen_lora_n500_sc8.jsonl",
        )

    def test_candidate_responses_prefers_all_sc_candidates(self):
        raw_row: dict[str, Any] = {
            "prediction": "selected",
            "all_responses": [
                {"content": "candidate 1"},
                {"content": "candidate 2"},
            ],
        }

        self.assertEqual(candidate_responses(raw_row), ["candidate 1", "candidate 2"])

    def test_candidate_responses_falls_back_to_prediction(self):
        self.assertEqual(
            candidate_responses({"prediction": "single prediction"}),
            ["single prediction"],
        )

    def test_update_row_costs_uses_budget_and_query_volumes(self):
        prices = {
            "self_hosted": {
                "qwen_2_5_1_5b": {
                    "derived_per_1m_tokens_usd": 0.18,
                }
            },
            "openai": {},
        }
        row = {
            "budget": "4",
            "train_cost_usd": "0.112",
            "judge_model": "",
            "judge_input_tokens": "0",
            "judge_output_tokens": "0",
        }
        raw_rows = [
            {
                "question": "What is 1 + 1?",
                "all_responses": [
                    {"content": "#### 2"},
                    {"content": "#### 2"},
                ],
            }
        ]

        updated = update_row_costs(
            row=row,
            raw_rows=raw_rows,
            prices=prices,
            counter=FixedTokenCounter(),
            query_counts=(1_000,),
        )

        self.assertEqual(updated["model_tokens_per_sample"], 20)
        self.assertEqual(updated["token_estimation_method"], "fixed_test_counter")
        self.assertAlmostEqual(
            updated["inference_cost_per_query_usd"],
            4 * 20 / 1_000_000 * 0.18,
        )
        self.assertAlmostEqual(
            updated["total_cost_usd_at_1000"],
            0.112 + 1_000 * 4 * 20 / 1_000_000 * 0.18,
        )

    def test_estimate_bon_judge_tokens_counts_each_candidate(self):
        raw_rows = [
            {
                "question": "What is 1 + 1?",
                "all_responses": [
                    {"content": "#### 2"},
                    {"content": "#### 3"},
                ],
            }
        ]

        input_tokens, output_tokens = estimate_bon_judge_tokens_per_query(
            raw_rows,
            counter=FixedTokenCounter(),
            judge_output_tokens_per_score=8,
        )

        self.assertEqual(input_tokens, 20)
        self.assertEqual(output_tokens, 16)


if __name__ == "__main__":
    unittest.main()
