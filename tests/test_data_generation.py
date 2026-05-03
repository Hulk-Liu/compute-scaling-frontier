import unittest

import pandas as pd

from src.data_generation import sample_gsm8k_rows, to_training_records


class DataGenerationTests(unittest.TestCase):
    def test_sample_gsm8k_rows_keeps_flow_inputs(self):
        dataset = [
            {"question": "q0", "answer": "#### 0"},
            {"question": "q1", "answer": "#### 1"},
            {"question": "q2", "answer": "#### 2"},
        ]

        df = sample_gsm8k_rows(dataset, n=2, seed=7)

        self.assertEqual(list(df.columns), ["source_index", "question", "answer"])
        self.assertEqual(len(df), 2)

    def test_to_training_records_outputs_chat_template(self):
        flow_output = pd.DataFrame(
            [
                {
                    "source_index": 42,
                    "question": "What is 7 + 5?",
                    "answer": "#### 12",
                    "teacher_response_content": "7 + 5 = 12.\n#### 12",
                }
            ]
        )

        records = to_training_records(flow_output)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["id"], "gsm8k-train-42")
        self.assertEqual(record["gold_answer"], "#### 12")
        self.assertEqual(record["messages"][0]["role"], "user")
        self.assertEqual(record["messages"][1]["role"], "assistant")
        self.assertIn("#### 12", record["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
