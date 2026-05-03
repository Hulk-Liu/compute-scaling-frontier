import unittest

from src.evaluator import (
    build_answer_diagnostics,
    canonicalize_number,
    evaluate_answer,
    evaluate_batch,
    extract_final_marker_answer,
    extract_gsm8k_gold_answer,
    extract_last_number,
)


class EvaluatorTests(unittest.TestCase):
    def test_canonicalize_number(self):
        self.assertEqual(canonicalize_number("1,200.0"), "1200")
        self.assertEqual(canonicalize_number("$-4.50"), "-4.5")
        self.assertEqual(canonicalize_number("-0.0"), "0")
        self.assertIsNone(canonicalize_number("not-a-number"))

    def test_extract_last_number_from_model_output(self):
        text = "We first compute 7 + 5 = 12. Therefore the answer is 12."
        self.assertEqual(extract_last_number(text), "12")

    def test_extract_last_number_handles_commas_and_decimals(self):
        text = "Revenue was $1,200.50, then the final answer is 2,401.00."
        self.assertEqual(extract_last_number(text), "2401")

    def test_extract_gsm8k_gold_answer_prefers_marker(self):
        answer = "She counts 2 boxes with 6 apples each. #### 12"
        self.assertEqual(extract_gsm8k_gold_answer(answer), "12")

    def test_evaluate_answer_exact_match(self):
        record = evaluate_answer(
            prediction="The result is 12.",
            gold_answer="She counts 2 boxes with 6 apples each. #### 12",
        )
        self.assertTrue(record.is_correct)
        self.assertEqual(record.extracted_prediction, "12")
        self.assertEqual(record.extracted_gold, "12")

    def test_evaluate_answer_incorrect_when_no_prediction_number(self):
        record = evaluate_answer(
            prediction="I cannot solve it.",
            gold_answer="She counts 2 boxes with 6 apples each. #### 12",
        )
        self.assertFalse(record.is_correct)
        self.assertIsNone(record.extracted_prediction)

    def test_extract_final_marker_answer_uses_number_after_marker(self):
        self.assertEqual(extract_final_marker_answer("13 appears first\n#### 42"), "42")
        self.assertIsNone(extract_final_marker_answer("13 appears but no marker"))

    def test_build_answer_diagnostics_detects_valid_marker_answer(self):
        diagnostics = build_answer_diagnostics(
            prediction="Reasoning has 13 first.\n#### 42",
            gold_answer="gold reasoning\n#### 42",
        )

        self.assertEqual(diagnostics["extracted_prediction"], "42")
        self.assertEqual(diagnostics["extracted_gold"], "42")
        self.assertTrue(diagnostics["is_correct"])
        self.assertTrue(diagnostics["has_final_marker"])
        self.assertEqual(diagnostics["extracted_marker_answer"], "42")
        self.assertTrue(diagnostics["answer_format_ok"])

    def test_build_answer_diagnostics_flags_truncated_marker(self):
        diagnostics = build_answer_diagnostics(
            prediction="The total is 57500.\n####",
            gold_answer="gold reasoning\n#### 57500",
        )

        self.assertEqual(diagnostics["extracted_prediction"], "57500")
        self.assertTrue(diagnostics["is_correct"])
        self.assertTrue(diagnostics["has_final_marker"])
        self.assertIsNone(diagnostics["extracted_marker_answer"])
        self.assertFalse(diagnostics["answer_format_ok"])

    def test_evaluate_batch_uses_strict_alignment(self):
        records, summary = evaluate_batch(
            predictions=["Answer: 12", "Answer: 10"],
            gold_answers=["#### 12", "#### 11"],
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.correct, 1)
        self.assertEqual(summary.accuracy, 0.5)

    def test_evaluate_batch_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_batch(predictions=["Answer: 12"], gold_answers=["#### 12", "#### 13"])


if __name__ == "__main__":
    unittest.main()
