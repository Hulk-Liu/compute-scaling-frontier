"""GSM8K-style exact-match evaluation utilities.

The project evaluates math answers by extracting the final numeric answer from
model text and comparing it to the canonical GSM8K gold answer after `####`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Iterable


NUMBER_PATTERN = re.compile(
    r"(?<![\w/])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w/])"
)
GSM8K_FINAL_ANSWER_MARKER = "####"


@dataclass(frozen=True)
class EvaluationRecord:
    """Per-example exact-match result."""

    prediction: str
    gold: str
    extracted_prediction: str | None
    extracted_gold: str | None
    is_correct: bool


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate exact-match result."""

    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def canonicalize_number(text: str) -> str | None:
    """Convert a numeric string into a stable comparison key.

    Examples:
        "1,200.0" -> "1200"
        "$-4.50" -> "-4.5"
    """

    cleaned = text.strip().replace(",", "")
    cleaned = cleaned.replace("$", "").replace("%", "")
    if not cleaned:
        return None

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None

    if value == 0:
        return "0"

    normalized = value.normalize()
    return format(normalized, "f")


def extract_last_number(text: str) -> str | None:
    """Extract the last numeric token from free-form model output."""

    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return canonicalize_number(matches[-1])


def extract_gsm8k_gold_answer(answer: str) -> str | None:
    """Extract the GSM8K final answer after `####`, falling back to last number."""

    if GSM8K_FINAL_ANSWER_MARKER in answer:
        answer = answer.rsplit(GSM8K_FINAL_ANSWER_MARKER, maxsplit=1)[-1]
    return extract_last_number(answer)


def evaluate_answer(prediction: str, gold_answer: str) -> EvaluationRecord:
    """Evaluate one model prediction against one GSM8K answer string."""

    extracted_prediction = extract_last_number(prediction)
    extracted_gold = extract_gsm8k_gold_answer(gold_answer)
    is_correct = (
        extracted_prediction is not None
        and extracted_gold is not None
        and extracted_prediction == extracted_gold
    )
    return EvaluationRecord(
        prediction=prediction,
        gold=gold_answer,
        extracted_prediction=extracted_prediction,
        extracted_gold=extracted_gold,
        is_correct=is_correct,
    )


def summarize(records: Iterable[EvaluationRecord]) -> EvaluationSummary:
    """Aggregate per-example evaluation records."""

    materialized = list(records)
    return EvaluationSummary(
        total=len(materialized),
        correct=sum(record.is_correct for record in materialized),
    )


def evaluate_batch(
    predictions: Iterable[str],
    gold_answers: Iterable[str],
) -> tuple[list[EvaluationRecord], EvaluationSummary]:
    """Evaluate aligned prediction/gold iterables."""

    records = [
        evaluate_answer(prediction, gold_answer)
        for prediction, gold_answer in zip(predictions, gold_answers, strict=True)
    ]
    return records, summarize(records)
