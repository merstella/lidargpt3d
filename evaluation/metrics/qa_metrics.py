import re
import string
from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


YES_SYNONYMS = {"yes", "yeah", "yep", "y", "true"}
NO_SYNONYMS = {"no", "nope", "n", "false"}
CANONICAL_SYNONYMS = {
    "bike": "bicycle",
    "bikes": "bicycle",
    "bicycles": "bicycle",
    "pedestrians": "pedestrian",
    "people": "pedestrian",
    "cars": "car",
    "vehicles": "vehicle",
    "trafficlight": "traffic light",
}


def _normalize_number_phrase(text: str) -> str:
    tokens = text.split()
    converted = []
    buffer = []

    def flush_buffer():
        if not buffer:
            return
        if all(token in NUMBER_WORDS or token == "and" for token in buffer):
            total = 0
            current = 0
            for token in buffer:
                if token == "and":
                    continue
                value = NUMBER_WORDS[token]
                if value == 100:
                    current = max(current, 1) * value
                else:
                    current += value
            total += current
            converted.append(str(total))
        else:
            converted.extend(buffer)
        buffer.clear()

    for token in tokens:
        if token in NUMBER_WORDS or token == "and":
            buffer.append(token)
        else:
            flush_buffer()
            converted.append(token)
    flush_buffer()
    return " ".join(converted)


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("-", " ")
    text = _normalize_number_phrase(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    if text in YES_SYNONYMS:
        return "yes"
    if text in NO_SYNONYMS:
        return "no"

    tokens = [CANONICAL_SYNONYMS.get(token, token) for token in text.split()]
    text = " ".join(tokens)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _group_accuracy(predictions: Sequence[Dict[str, object]], key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, list] = OrderedDict()
    for item in predictions:
        bucket = str(item.get(key, "unknown"))
        buckets.setdefault(bucket, []).append(float(item["is_correct"]))

    breakdown = OrderedDict()
    for bucket, scores in buckets.items():
        breakdown[bucket] = {
            "accuracy": float(np.mean(scores) * 100.0) if scores else 0.0,
            "num_samples": len(scores),
        }
    return breakdown


def evaluate_qa_predictions(predictions: Sequence[Dict[str, object]]) -> Dict[str, object]:
    scored_predictions = []
    for item in predictions:
        score = max(exact_match(item["prediction"], ref) for ref in item.get("references", [item["ground_truth"]]))
        enriched = dict(item)
        enriched["normalized_prediction"] = normalize_answer(item["prediction"])
        enriched["normalized_ground_truth"] = normalize_answer(item["ground_truth"])
        enriched["is_correct"] = score
        scored_predictions.append(enriched)

    overall_scores = [float(item["is_correct"]) for item in scored_predictions]
    overall_accuracy = float(np.mean(overall_scores) * 100.0) if overall_scores else 0.0

    return {
        "num_questions": len(scored_predictions),
        "exact_match": overall_accuracy,
        "accuracy": overall_accuracy,
        "accuracy_by_question_type": _group_accuracy(scored_predictions, "question_type"),
        "accuracy_by_answer_type": _group_accuracy(scored_predictions, "answer_type"),
        "scored_predictions": scored_predictions,
    }

