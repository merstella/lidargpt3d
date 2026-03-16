import argparse
import json
import os
import re
import string
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from minigpt4.common.eval_utils import init_model, prepare_texts
from minigpt4.conversation.conversation import CONV_VISION

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    NuScenes = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    SmoothingFunction = None
    sentence_bleu = None
    meteor_score = None

try:
    from rouge import Rouge
except ImportError:
    Rouge = None


CAPTION_DEFAULT_PROMPT = "Describe this driving scene in detail."


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def load_annotations(anno_path):
    if anno_path.endswith(".jsonl"):
        items = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return data["annotations"]
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
        raise ValueError("JSON annotation must be a list or contain `annotations`/`results` list.")
    return data


def _clean_prompt(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("<point>", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_ref_list(value):
    if value is None:
        return [""]
    if isinstance(value, list):
        refs = [str(v).strip() for v in value if str(v).strip()]
        return refs if refs else [""]
    ref = str(value).strip()
    return [ref]


def extract_prompt_and_refs(record, task_type):
    convs = record.get("conversations", None)

    question = record.get("question", None)
    answer = record.get("answer", None)
    caption = record.get("caption", None)

    if convs and isinstance(convs, list):
        if question is None and len(convs) >= 1:
            question = convs[0].get("value", "")
        if answer is None and len(convs) >= 2:
            answer = convs[1].get("value", "")
        if caption is None and len(convs) >= 2:
            caption = convs[1].get("value", "")

    if task_type == "nuscenesqa":
        prompt = _clean_prompt(question)
        refs = _to_ref_list(answer)
    else:
        prompt = _clean_prompt(question) if question is not None else ""
        if not prompt:
            prompt = CAPTION_DEFAULT_PROMPT
        refs = _to_ref_list(caption if caption is not None else answer)

    return prompt, refs


def extract_sample_token(record):
    token = (
        record.get("sample_token")
        or record.get("token")
        or record.get("sample", {}).get("token")
    )
    if token is None:
        raise KeyError("Missing `sample_token` in annotation record.")
    return token


class NuScenesEvalDataset(Dataset):
    def __init__(
        self,
        records,
        nusc,
        task_type,
        pointnum=8192,
        normalize_pc=False,
        use_intensity=True,
        use_time_or_ring=True,
        return_dim=6,
        feat_mode="const_0p4",
    ):
        self.records = records
        self.nusc = nusc
        self.task_type = task_type
        self.pointnum = pointnum
        self.normalize_pc = normalize_pc
        self.use_intensity = use_intensity
        self.use_time_or_ring = use_time_or_ring
        self.return_dim = return_dim
        self.feat_mode = feat_mode
        self._lidar_path_cache = {}
        self._validate_config()

    def __len__(self):
        return len(self.records)

    def _get_lidar_path(self, sample_token):
        if sample_token in self._lidar_path_cache:
            return self._lidar_path_cache[sample_token]

        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, sd["filename"])
        self._lidar_path_cache[sample_token] = lidar_path
        return lidar_path

    @staticmethod
    def _read_lidar_bin(lidar_path):
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size % 5 != 0:
            raise ValueError(f"Unexpected lidar format for {lidar_path}, raw.size % 5 != 0")
        return raw.reshape(-1, 5)

    def _build_xyz_feat(self, pts5):
        xyz = pts5[:, :3].astype(np.float32)
        if self.feat_mode == "const_0p4":
            feat = np.full((xyz.shape[0], 3), 0.4, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported feat_mode: {self.feat_mode}")
        return xyz, feat

    def _sample_points(self, xyz, feat):
        n = xyz.shape[0]
        if n >= self.pointnum:
            idx = np.random.choice(n, self.pointnum, replace=False)
        else:
            idx = np.random.choice(n, self.pointnum, replace=True)
        return xyz[idx], feat[idx]

    @staticmethod
    def _pc_norm(xyz):
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        scale = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if scale > 0:
            xyz = xyz / scale
        return xyz

    def _validate_config(self):
        if self.return_dim != 6:
            raise ValueError("Only return_dim=6 is supported in Uni3D-style eval mode.")
        if self.feat_mode not in {"const_0p4"}:
            raise ValueError(f"Unsupported feat_mode `{self.feat_mode}`.")

    def __getitem__(self, index):
        record = self.records[index]
        sample_token = extract_sample_token(record)
        prompt, refs = extract_prompt_and_refs(record, self.task_type)

        lidar_path = self._get_lidar_path(sample_token)
        pts5 = self._read_lidar_bin(lidar_path)
        xyz, feat = self._build_xyz_feat(pts5)
        xyz, feat = self._sample_points(xyz, feat)
        if self.normalize_pc:
            xyz = self._pc_norm(xyz)

        return {
            "xyz": torch.from_numpy(xyz.astype(np.float32)),
            "feat": torch.from_numpy(feat.astype(np.float32)),
            "sample_token": sample_token,
            "question": prompt,
            "ground_truths": refs,
        }


def eval_collate(batch):
    xyz = torch.stack([x["xyz"] for x in batch], dim=0)
    feat = torch.stack([x["feat"] for x in batch], dim=0)
    return {
        "xyz": xyz,
        "feat": feat,
        "sample_token": [x["sample_token"] for x in batch],
        "question": [x["question"] for x in batch],
        "ground_truths": [x["ground_truths"] for x in batch],
    }


def _postprocess_generation(text):
    text = text.lower().replace("<unk>", "").strip()
    text = text.split("###")[0]
    text = text.split("assistant:")[-1].strip()
    return text


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_refs(metric_fn, prediction, refs):
    return max(metric_fn(prediction, ref) for ref in refs) if refs else metric_fn(prediction, "")


def evaluate_nuscenesqa(results):
    em_scores = []
    f1_scores = []
    enriched = []

    for item in results:
        pred = item["model_output"]
        refs = item["ground_truths"]
        em = best_over_refs(exact_match_score, pred, refs)
        f1 = best_over_refs(f1_score, pred, refs)
        em_scores.append(em)
        f1_scores.append(f1)

        item = dict(item)
        item["exact_match"] = em
        item["f1"] = f1
        enriched.append(item)

    metrics = {
        "exact_match": 100.0 * float(np.mean(em_scores)) if em_scores else 0.0,
        "f1": 100.0 * float(np.mean(f1_scores)) if f1_scores else 0.0,
        "num_samples": len(results),
    }
    return metrics, enriched


def _caption_metrics_single(pred, refs, rouge_eval, smoothing_fn, meteor_enabled):
    if not refs:
        refs = [""]
    ref_tokens = [r.split() for r in refs]
    pred_tokens = pred.split()

    scores = {
        "bleu-1": 0.0,
        "bleu-2": 0.0,
        "bleu-3": 0.0,
        "bleu-4": 0.0,
        "rouge-1": 0.0,
        "rouge-2": 0.0,
        "rouge-l": 0.0,
        "meteor": 0.0,
    }

    if sentence_bleu is not None and smoothing_fn is not None:
        scores["bleu-1"] = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_fn) * 100
        scores["bleu-2"] = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_fn) * 100
        scores["bleu-3"] = sentence_bleu(ref_tokens, pred_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing_fn) * 100
        scores["bleu-4"] = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_fn) * 100

    if rouge_eval is not None:
        best_r1 = best_r2 = best_rl = 0.0
        for ref in refs:
            try:
                rouge_scores = rouge_eval.get_scores(pred or " ", ref or " ")[0]
                best_r1 = max(best_r1, rouge_scores["rouge-1"]["f"] * 100)
                best_r2 = max(best_r2, rouge_scores["rouge-2"]["f"] * 100)
                best_rl = max(best_rl, rouge_scores["rouge-l"]["f"] * 100)
            except ValueError:
                continue
        scores["rouge-1"] = best_r1
        scores["rouge-2"] = best_r2
        scores["rouge-l"] = best_rl

    if meteor_enabled and meteor_score is not None:
        best_meteor = 0.0
        for ref in refs:
            try:
                cur = meteor_score([ref.split()], pred.split()) * 100
                best_meteor = max(best_meteor, cur)
            except Exception:
                continue
        scores["meteor"] = best_meteor

    return scores


def evaluate_nucaption(results):
    rouge_eval = Rouge() if Rouge is not None else None
    smoothing_fn = SmoothingFunction().method1 if SmoothingFunction is not None else None
    meteor_enabled = meteor_score is not None

    metric_buckets = {
        "bleu-1": [],
        "bleu-2": [],
        "bleu-3": [],
        "bleu-4": [],
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "meteor": [],
    }

    enriched = []
    for item in results:
        pred = item["model_output"]
        refs = item["ground_truths"]
        scores = _caption_metrics_single(pred, refs, rouge_eval, smoothing_fn, meteor_enabled)
        for k, v in scores.items():
            metric_buckets[k].append(v)

        enriched_item = dict(item)
        enriched_item["scores"] = scores
        enriched.append(enriched_item)

    metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in metric_buckets.items()}
    metrics["num_samples"] = len(results)
    return metrics, enriched


def generate_predictions(model, dataloader, args):
    conv_temp = CONV_VISION.copy()
    conv_temp.system = ""

    results = {
        "task_type": args.task_type,
        "results": [],
    }

    for batch in tqdm(dataloader, desc="Generating"):
        xyz = batch["xyz"].cuda()
        feat = batch["feat"].cuda()
        questions = [q if q else CAPTION_DEFAULT_PROMPT for q in batch["question"]]
        prompts = prepare_texts(questions, conv_temp)

        model.eval()
        with torch.inference_mode():
            answers = model.generate(
                texts=prompts,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                min_length=args.min_length,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                length_penalty=args.length_penalty,
                temperature=args.temperature,
                do_sample=args.do_sample,
                xyz=xyz,
                feat=feat,
            )

        for sample_token, question, refs, output in zip(
            batch["sample_token"], questions, batch["ground_truths"], answers
        ):
            results["results"].append(
                {
                    "sample_token": sample_token,
                    "question": question,
                    "ground_truths": refs,
                    "model_output": _postprocess_generation(output),
                }
            )

    return results


def run_evaluation(predictions, task_type):
    if task_type == "nuscenesqa":
        return evaluate_nuscenesqa(predictions["results"])
    if task_type == "nucaption":
        return evaluate_nucaption(predictions["results"])
    raise ValueError(f"Unsupported task_type: {task_type}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MiniGPT-3D on NuScenesQA / NuCaption")
    parser.add_argument("--cfg-path", type=str, default="./eval_configs/benchmark_evaluation_nuscenes.yaml")
    parser.add_argument("--out_path", type=str, default="./output/nuscenes_eval")
    parser.add_argument("--anno_path", type=str, required=True, help="Path to NuScenesQA/NuCaption annotations (.json or .jsonl)")
    parser.add_argument("--task_type", type=str, choices=["nuscenesqa", "nucaption"], required=True)
    parser.add_argument("--data_root", type=str, required=True, help="nuScenes dataroot")
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes version")

    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--normalize_pc", type=str2bool, default=False)
    parser.add_argument("--use_intensity", type=str2bool, default=True)
    parser.add_argument("--use_time_or_ring", type=str2bool, default=True)
    parser.add_argument("--return_dim", type=int, default=6)
    parser.add_argument("--feat_mode", type=str, default="const_0p4")

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--start_eval", action="store_true", default=False, help="Run metric evaluation after generation")
    parser.add_argument("--force_regen", action="store_true", default=False, help="Regenerate predictions even if file exists")

    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do_sample", type=str2bool, default=True)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override settings in key=value format for model config",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if NuScenes is None:
        raise ImportError(
            "nuscenes-devkit is required. Install it with `pip install nuscenes-devkit`."
        )

    os.makedirs(args.out_path, exist_ok=True)
    anno_name = os.path.splitext(os.path.basename(args.anno_path))[0]
    pred_file = os.path.join(args.out_path, f"{anno_name}_{args.task_type}_predictions.json")
    eval_file = os.path.join(args.out_path, f"{anno_name}_{args.task_type}_evaluated.json")

    if not os.path.exists(pred_file) or args.force_regen:
        print("[INFO] Loading annotations...")
        records = load_annotations(args.anno_path)
        print(f"[INFO] Loaded {len(records)} records.")

        print("[INFO] Building nuScenes devkit...")
        nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
        dataset = NuScenesEvalDataset(
            records=records,
            nusc=nusc,
            task_type=args.task_type,
            pointnum=args.pointnum,
            normalize_pc=args.normalize_pc,
            use_intensity=args.use_intensity,
            use_time_or_ring=args.use_time_or_ring,
            return_dim=args.return_dim,
            feat_mode=args.feat_mode,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_collate,
        )

        print("[INFO] Initializing model...")
        model = init_model(args).eval()
        print("[INFO] Starting generation...")
        predictions = generate_predictions(model, dataloader, args)

        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved predictions to {pred_file}")

        del model
        torch.cuda.empty_cache()
    else:
        print(f"[INFO] Found existing predictions: {pred_file}")
        with open(pred_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    if args.start_eval:
        metrics, enriched_results = run_evaluation(predictions, args.task_type)
        eval_payload = {
            "task_type": args.task_type,
            "metrics": metrics,
            "results": enriched_results,
        }
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved evaluation to {eval_file}")
        print("[INFO] Metrics:")
        for key, value in metrics.items():
            if key == "num_samples":
                print(f"  - {key}: {value}")
            else:
                print(f"  - {key}: {value:.4f}")


if __name__ == "__main__":
    main()
