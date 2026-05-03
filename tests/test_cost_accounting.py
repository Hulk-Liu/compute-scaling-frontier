import unittest

from src.cost_accounting import (
    inference_cost_per_query_usd,
    openai_token_cost_usd,
    self_hosted_token_cost_usd,
    total_serving_cost_usd,
    training_cost_usd,
)


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


class CostAccountingTests(unittest.TestCase):
    def test_openai_token_cost(self):
        cost = openai_token_cost_usd(
            PRICES,
            model="gpt-4o-mini",
            input_tokens=1_000_000,
            output_tokens=2_000_000,
        )

        self.assertAlmostEqual(cost, 1.35)

    def test_self_hosted_token_cost(self):
        cost = self_hosted_token_cost_usd(
            PRICES,
            model_key="qwen_2_5_1_5b",
            tokens=1_000_000,
        )

        self.assertAlmostEqual(cost, 0.18)

    def test_training_cost_combines_data_generation_and_gpu(self):
        cost = training_cost_usd(
            PRICES,
            synthetic_examples=1_000,
            gpu_hours=2.0,
        )

        self.assertAlmostEqual(cost.data_generation_usd, 1.0)
        self.assertAlmostEqual(cost.gpu_usd, 0.8)
        self.assertAlmostEqual(cost.total_usd, 1.8)

    def test_inference_cost_without_judge(self):
        cost = inference_cost_per_query_usd(
            PRICES,
            budget=4,
            model_tokens_per_sample=250,
        )

        self.assertAlmostEqual(cost.model_usd, 0.00018)
        self.assertAlmostEqual(cost.judge_usd, 0.0)
        self.assertAlmostEqual(cost.total_usd, 0.00018)

    def test_inference_cost_with_judge(self):
        cost = inference_cost_per_query_usd(
            PRICES,
            budget=4,
            model_tokens_per_sample=250,
            judge_model="gpt-4o-mini",
            judge_input_tokens=1_000,
            judge_output_tokens=20,
        )

        self.assertAlmostEqual(cost.model_usd, 0.00018)
        self.assertAlmostEqual(cost.judge_usd, 0.000162)
        self.assertAlmostEqual(cost.total_usd, 0.000342)

    def test_total_serving_cost(self):
        total = total_serving_cost_usd(
            train_cost_usd=1.8,
            inference_cost_per_query_usd=0.00018,
            query_count=10_000,
        )

        self.assertAlmostEqual(total, 3.6)

    def test_rejects_negative_cost_inputs(self):
        with self.assertRaises(ValueError):
            training_cost_usd(PRICES, synthetic_examples=-1, gpu_hours=1.0)

        with self.assertRaises(ValueError):
            total_serving_cost_usd(
                train_cost_usd=0.0,
                inference_cost_per_query_usd=0.1,
                query_count=-1,
            )


if __name__ == "__main__":
    unittest.main()
