import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"\w+", str(text).lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(predictions: Sequence[str], references: Sequence[Sequence[str]], max_order: int = 4) -> Dict[str, float]:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize_text(pred)
        ref_tokens = [tokenize_text(ref) for ref in refs if ref]
        if not ref_tokens:
            ref_tokens = [[]]

        pred_length += len(pred_tokens)
        ref_length += len(min(ref_tokens, key=lambda ref: (abs(len(ref) - len(pred_tokens)), len(ref))))

        merged_ref_ngram_counts = Counter()
        for ref in ref_tokens:
            for order in range(1, max_order + 1):
                merged_ref_ngram_counts |= _ngrams(ref, order)

        for order in range(1, max_order + 1):
            pred_ngram_counts = _ngrams(pred_tokens, order)
            overlap = pred_ngram_counts & merged_ref_ngram_counts
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(pred_tokens) - order + 1, 0)

    precisions = [0.0] * max_order
    smooth = 1.0
    for order in range(max_order):
        if possible_matches_by_order[order] > 0:
            if matches_by_order[order] > 0:
                precisions[order] = matches_by_order[order] / possible_matches_by_order[order]
            else:
                smooth *= 2.0
                precisions[order] = 1.0 / (smooth * possible_matches_by_order[order])

    if pred_length == 0:
        brevity_penalty = 0.0
    elif pred_length > ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - float(ref_length) / max(pred_length, 1))

    bleu_scores = {}
    for order in range(1, max_order + 1):
        if min(precisions[:order]) <= 0:
            score = 0.0
        else:
            score = brevity_penalty * math.exp(sum(math.log(p) for p in precisions[:order]) / order)
        bleu_scores[f"bleu-{order}"] = score * 100.0
    return bleu_scores


def rouge_l_score(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_text(prediction)
    ref_tokens = tokenize_text(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ref_tokens) + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall)) * 100.0


class CiderScorer:
    def __init__(self, references: Sequence[Sequence[str]], n: int = 4):
        self.n = n
        self.num_docs = len(references)
        self.document_frequency = defaultdict(int)

        for refs in references:
            per_doc = set()
            for ref in refs:
                tokens = tokenize_text(ref)
                for order in range(1, self.n + 1):
                    per_doc.update(_ngrams(tokens, order).keys())
            for ngram in per_doc:
                self.document_frequency[ngram] += 1

    def _counts_to_tfidf(self, counts: Counter) -> Dict[Tuple[str, ...], float]:
        weights = {}
        for ngram, term_freq in counts.items():
            doc_freq = self.document_frequency.get(ngram, 1)
            idf = math.log(max(1.0, float(self.num_docs)) / doc_freq)
            weights[ngram] = float(term_freq) * idf
        return weights

    @staticmethod
    def _cosine_similarity(vec_a: Dict[Tuple[str, ...], float], vec_b: Dict[Tuple[str, ...], float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[key] * vec_b[key] for key in common_keys)
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def score(self, prediction: str, refs: Sequence[str]) -> float:
        pred_tokens = tokenize_text(prediction)
        scores = []

        for order in range(1, self.n + 1):
            pred_counts = _ngrams(pred_tokens, order)
            pred_vec = self._counts_to_tfidf(pred_counts)

            order_scores = []
            for ref in refs:
                ref_counts = _ngrams(tokenize_text(ref), order)
                ref_vec = self._counts_to_tfidf(ref_counts)
                order_scores.append(self._cosine_similarity(pred_vec, ref_vec))
            scores.append(sum(order_scores) / max(len(order_scores), 1))

        return 10.0 * float(np.mean(scores))


class BERTScoreScorer:
    def __init__(
        self,
        model_name_or_path: str = "./params_weight/bert-base-uncased",
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
        except OSError:
            fallback = "bert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModel.from_pretrained(fallback)

        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def _encode(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].bool()
        special_mask = torch.zeros_like(attention_mask)
        for special_id in self.tokenizer.all_special_ids:
            special_mask |= encoded["input_ids"].eq(special_id)
        valid_mask = attention_mask & (~special_mask)
        return hidden, valid_mask

    @staticmethod
    def _pair_score(
        pred_hidden: torch.Tensor,
        pred_mask: torch.Tensor,
        ref_hidden: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> Tuple[float, float, float]:
        pred_vec = pred_hidden[pred_mask]
        ref_vec = ref_hidden[ref_mask]
        if pred_vec.numel() == 0 or ref_vec.numel() == 0:
            return 0.0, 0.0, 0.0

        pred_vec = torch.nn.functional.normalize(pred_vec, p=2, dim=-1)
        ref_vec = torch.nn.functional.normalize(ref_vec, p=2, dim=-1)
        sim = pred_vec @ ref_vec.transpose(0, 1)
        precision = sim.max(dim=1).values.mean().item()
        recall = sim.max(dim=0).values.mean().item()
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision * 100.0, recall * 100.0, f1 * 100.0

    def score(self, predictions: Sequence[str], references: Sequence[Sequence[str]]) -> Dict[str, float]:
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for start in tqdm(range(0, len(predictions), self.batch_size), desc="BERTScore", leave=False):
            pred_batch = predictions[start : start + self.batch_size]
            ref_batch = references[start : start + self.batch_size]
            pred_hidden, pred_mask = self._encode(pred_batch)

            flat_refs = []
            ref_boundaries = []
            cursor = 0
            for refs in ref_batch:
                refs = list(refs) if refs else [""]
                flat_refs.extend(refs)
                ref_boundaries.append((cursor, cursor + len(refs)))
                cursor += len(refs)

            ref_hidden, ref_mask = self._encode(flat_refs)

            for batch_index, (left, right) in enumerate(ref_boundaries):
                best_precision = 0.0
                best_recall = 0.0
                best_f1 = 0.0
                for ref_index in range(left, right):
                    precision, recall, f1 = self._pair_score(
                        pred_hidden[batch_index],
                        pred_mask[batch_index],
                        ref_hidden[ref_index],
                        ref_mask[ref_index],
                    )
                    if f1 > best_f1:
                        best_precision = precision
                        best_recall = recall
                        best_f1 = f1
                precision_scores.append(best_precision)
                recall_scores.append(best_recall)
                f1_scores.append(best_f1)

        return {
            "bertscore_precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
            "bertscore_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
            "bertscore_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        }


def evaluate_caption_predictions(
    predictions: Sequence[Dict[str, object]],
    bert_model: str,
    bert_device: str,
    bert_batch_size: int,
) -> Dict[str, object]:
    pred_texts = [str(item["prediction"]) for item in predictions]
    references = [list(item.get("references") or [str(item["ground_truth"])]) for item in predictions]

    bleu_scores = corpus_bleu(pred_texts, references, max_order=4)

    rouge_l_scores = []
    for pred, refs in zip(pred_texts, references):
        rouge_l_scores.append(max(rouge_l_score(pred, ref) for ref in refs if ref) if refs else 0.0)

    cider = CiderScorer(references)
    cider_scores = [cider.score(pred, refs) for pred, refs in zip(pred_texts, references)]

    bert_scorer = BERTScoreScorer(
        model_name_or_path=bert_model,
        device=bert_device,
        batch_size=bert_batch_size,
    )
    bert_scores = bert_scorer.score(pred_texts, references)

    return {
        "num_scenes": len(predictions),
        "bleu-1": bleu_scores["bleu-1"],
        "bleu-4": bleu_scores["bleu-4"],
        "rouge-l": float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0,
        "cider": float(np.mean(cider_scores)) if cider_scores else 0.0,
        **bert_scores,
    }
