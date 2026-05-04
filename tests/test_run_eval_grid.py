import asyncio
import tempfile
import unittest
from pathlib import Path

from src.run_eval_grid import (
    ModelVariant,
    StrategySpec,
    cell_slug,
    default_model_variants,
    default_strategy_specs,
    run_eval_grid,
)
from src.run_its_experiment import write_jsonl


class RunEvalGridTests(unittest.TestCase):
    def test_default_grid_has_three_models_and_three_strategies(self):
        models = default_model_variants(
            base_model="base",
            lora_n100_model="n100",
            lora_n500_model="n500",
            train_gpu_hours_n100=0.03,
            train_gpu_hours_n500=0.125,
        )
        strategies = default_strategy_specs()

        self.assertEqual([model.train_size for model in models], [0, 100, 500])
        self.assertEqual(
            [(strategy.name, strategy.budget) for strategy in strategies],
            [("greedy", 1), ("sc", 4), ("sc", 8)],
        )

    def test_cell_slug_includes_model_variant_and_budget(self):
        model = ModelVariant(
            name="lora_n500",
            model_id="qwen-gsm8k-lora-n500",
            train_size=500,
            train_gpu_hours=0.125,
        )

        self.assertEqual(
            cell_slug(model, StrategySpec(name="greedy", budget=1)),
            "qwen_lora_n500_greedy",
        )
        self.assertEqual(
            cell_slug(model, StrategySpec(name="sc", budget=8)),
            "qwen_lora_n500_sc8",
        )

    def test_resume_raw_file_shape_matches_aggregate_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path = Path(tmp_dir) / "qwen_base_greedy.jsonl"
            write_jsonl(
                [
                    {
                        "id": "gsm8k-test-1",
                        "source_index": 1,
                        "question": "q",
                        "answer": "#### 3",
                        "prediction": "#### 3",
                        "model": "base",
                        "strategy": "greedy",
                        "budget": 1,
                        "raw_response": {"content": "#### 3"},
                    }
                ],
                raw_path,
            )

            self.assertEqual(raw_path.read_text(encoding="utf-8").count("\n"), 1)

    def test_dry_run_does_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "aggregated.csv"
            raw_dir = Path(tmp_dir) / "raw"

            rows = asyncio.run(
                run_eval_grid(
                    eval_rows=[],
                    models=[
                        ModelVariant(
                            name="base",
                            model_id="base",
                            train_size=0,
                            train_gpu_hours=0.0,
                        )
                    ],
                    strategies=[StrategySpec(name="greedy", budget=1)],
                    endpoint="http://127.0.0.1:8000/v1",
                    api_key="unused",
                    raw_dir=raw_dir,
                    prices={},
                    output_csv=output,
                    model_tokens_per_sample=0,
                    max_tokens=16,
                    resume=False,
                    dry_run=True,
                )
            )

            self.assertEqual(rows, [])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
