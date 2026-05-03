import unittest

from src.validate_training_data import validate_record, validate_records


class ValidateTrainingDataTests(unittest.TestCase):
    def test_validate_record_matches_teacher_response_to_gold(self):
        evaluation = validate_record(
            {
                "teacher_response": "7 + 5 = 12.\n#### 12",
                "gold_answer": "#### 12",
            },
            row_index=1,
        )

        self.assertTrue(evaluation.is_correct)

    def test_validate_records_returns_accuracy(self):
        evaluations, accuracy = validate_records(
            [
                {"teacher_response": "#### 12", "gold_answer": "#### 12"},
                {"teacher_response": "#### 9", "gold_answer": "#### 10"},
            ]
        )

        self.assertEqual(len(evaluations), 2)
        self.assertEqual(accuracy, 0.5)

    def test_validate_record_requires_teacher_response(self):
        with self.assertRaises(ValueError):
            validate_record({"gold_answer": "#### 12"}, row_index=1)


if __name__ == "__main__":
    unittest.main()
