import tempfile
import unittest
from pathlib import Path

from src.run_its_experiment import (
    _base_raw_row,
    _response_content,
    build_math_prompt,
    final_answer_projection,
    read_eval_records,
    write_jsonl,
)


class RunITSExperimentTests(unittest.TestCase):
    def test_build_math_prompt_requests_final_marker(self):
        prompt = build_math_prompt("What is 7 + 5?")

        self.assertIn("What is 7 + 5?", prompt)
        self.assertIn("#### <number>", prompt)

    def test_response_content_handles_none(self):
        self.assertEqual(_response_content({"content": None}), "")
        self.assertEqual(_response_content({"content": "ok"}), "ok")

    def test_final_answer_projection_votes_on_extracted_number(self):
        self.assertEqual(
            final_answer_projection("One path ends with #### 1,200.0"),
            "1200",
        )

    def test_final_answer_projection_falls_back_to_stripped_text(self):
        self.assertEqual(final_answer_projection(" no numeric answer "), "no numeric answer")

    def test_base_raw_row_sets_common_schema(self):
        row = _base_raw_row(
            eval_row={
                "id": "gsm8k-test-1",
                "source_index": 1,
                "question": "q",
                "answer": "#### 1",
            },
            prediction="#### 1",
            model="gpt-4o-mini",
            strategy="sc",
            budget=4,
            raw_response={"content": "#### 1"},
        )

        self.assertEqual(row["id"], "gsm8k-test-1")
        self.assertEqual(row["prediction"], "#### 1")
        self.assertEqual(row["strategy"], "sc")
        self.assertEqual(row["budget"], 4)
        self.assertEqual(row["raw_response"], {"content": "#### 1"})

    def test_read_eval_records_respects_n_eval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "eval.jsonl"
            write_jsonl(
                [
                    {"id": "1", "source_index": 1, "question": "q1", "answer": "#### 1"},
                    {"id": "2", "source_index": 2, "question": "q2", "answer": "#### 2"},
                ],
                path,
            )

            rows = read_eval_records(path, n_eval=1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "1")

    def test_write_jsonl_writes_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "raw.jsonl"
            write_jsonl([{"id": "1", "prediction": "#### 1"}], path)

            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
