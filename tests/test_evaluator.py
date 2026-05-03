import unittest

from src.evaluator import (
    canonicalize_number,
    evaluate_answer,
    evaluate_batch,
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
