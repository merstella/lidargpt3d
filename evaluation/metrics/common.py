import hashlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
except ImportError:
    NuScenes = None
    LidarPointCloud = None


SUPPORTED_CACHE_EXTENSIONS = (".pt", ".pth", ".npy", ".npz")
CAPTION_PROMPT = "Describe the current driving scene."
QA_PROMPT_TEMPLATE = "<scene>\nQuestion: {question}\nAnswer:"


@dataclass
class CaptionSceneRecord:
    scene_id: str
    sample_token: str
    prompt: str
    ground_truth: str
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QARecord:
    scene_id: str
    sample_token: str
    question_id: str
    question: str
    prompt: str
    ground_truth: str
    question_type: str
    answer_type: str
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def load_json_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {file_path}")

    if file_path.suffix == ".jsonl":
        records = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("annotations", "results", "questions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        f"Unsupported annotation structure in {file_path}. Expected list or dict with list payload."
    )


def _get_nested(record: Dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_text_from_conversations(record: Dict[str, Any], role: str) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") == role:
            return clean_text(turn.get("value", ""))
    return ""


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("<point>", " ").replace("<scene>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sample_token(record: Dict[str, Any]) -> str:
    token = (
        record.get("sample_token")
        or record.get("scene_id")
        or record.get("scene_token")
        or record.get("token")
        or _get_nested(record, "sample", "token")
        or _get_nested(record, "meta", "sample_token")
    )
    if token is None:
        raise KeyError("Missing sample token / scene key in annotation record.")
    return str(token)


def extract_caption_ground_truth(record: Dict[str, Any]) -> str:
    for key in ("answer_lidar", "ground_truth", "caption"):
        value = record.get(key)
        if value:
            return clean_text(value)
    text = _extract_text_from_conversations(record, "assistant")
    return clean_text(text)


def extract_qa_ground_truth(record: Dict[str, Any]) -> str:
    for key in ("ground_truth", "answer"):
        value = record.get(key)
        if value:
            return clean_text(value)
    text = _extract_text_from_conversations(record, "assistant")
    return clean_text(text)


def extract_question(record: Dict[str, Any]) -> str:
    if record.get("question"):
        return clean_text(record["question"])
    text = _extract_text_from_conversations(record, "human")
    return clean_text(text)


def parse_key_value_list(items: Optional[Sequence[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = OrderedDict()
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_ablation_tags(items: Optional[Sequence[str]]) -> Dict[str, str]:
    parsed: Dict[str, str] = OrderedDict()
    for item in items or []:
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key.strip()] = value.strip()
        else:
            parsed[item.strip()] = "true"
    return parsed


def stage_sort_key(stage_name: str) -> Tuple[int, str]:
    match = re.search(r"stage[_-]?(\d+)", stage_name.lower())
    if match:
        return int(match.group(1)), stage_name
    return 10**9, stage_name


def build_run_specs(
    stage_name: str,
    checkpoint: Optional[str],
    second_checkpoint: Optional[str],
    checkpoint_spec: Optional[Sequence[str]],
    second_checkpoint_spec: Optional[Sequence[str]],
    allowed_stages: Optional[Sequence[str]] = None,
) -> List[Dict[str, Optional[str]]]:
    run_specs: List[Dict[str, Optional[str]]] = []
    ckpt_map = parse_key_value_list(checkpoint_spec)
    second_map = parse_key_value_list(second_checkpoint_spec)
    allowed = {stage.lower() for stage in allowed_stages or []}

    if ckpt_map:
        stage_names = sorted(ckpt_map.keys(), key=stage_sort_key)
        for name in stage_names:
            if allowed and name.lower() not in allowed:
                continue
            run_specs.append(
                {
                    "stage_name": name,
                    "checkpoint": ckpt_map[name],
                    "second_checkpoint": second_map.get(name),
                }
            )
        if not run_specs:
            raise ValueError("No run specs remain after stage filtering.")
        return run_specs

    if allowed and stage_name.lower() not in allowed:
        raise ValueError(
            f"Stage `{stage_name}` is not valid for this script. Allowed: {sorted(allowed)}"
        )
    return [
        {
            "stage_name": stage_name,
            "checkpoint": checkpoint,
            "second_checkpoint": second_checkpoint,
        }
    ]


def _stable_seed(key: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def postprocess_generation(text: str) -> str:
    text = str(text).replace("<unk>", " ").strip()
    for token in ("###", "</s>"):
        text = text.split(token)[0]
    text = text.split("Assistant:")[-1]
    text = text.split("[/INST]")[-1]
    return re.sub(r"\s+", " ", text).strip()


def build_generation_prompts(texts: Sequence[str]) -> List[str]:
    return [
        f"<s>[INST] <PC><PointCloudHere></PC> <s>[INST] {text} [/INST]"
        for text in texts
    ]


def init_model_for_eval(
    cfg_path: str,
    device: str,
    checkpoint: Optional[str] = None,
    second_checkpoint: Optional[str] = None,
    cfg_options: Optional[Sequence[str]] = None,
):
    import minigpt4  # noqa: F401
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry

    options = list(cfg_options or [])
    if checkpoint is not None or second_checkpoint is not None:
        options.append(f"model.ckpt={checkpoint or ''}")
        options.append(f"model.second_ckpt={second_checkpoint or ''}")

    args = SimpleNamespace(cfg_path=cfg_path, options=options)
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    model = model.to(device)
    model.eval()
    return model, cfg


class SceneCacheLoader:
    def __init__(
        self,
        scene_cache: str,
        pointnum: int = 8192,
        normalize_pc: bool = False,
        feat_mode: str = "const_0p4",
        scan_cache_on_miss: bool = False,
        nusc_root: Optional[str] = None,
        nusc_version: str = "v1.0-trainval",
        num_sweeps: int = 1,
        seed: int = 42,
    ):
        self.scene_cache = Path(scene_cache) if scene_cache else None
        self.pointnum = pointnum
        self.normalize_pc = normalize_pc
        self.feat_mode = feat_mode
        self.scan_cache_on_miss = scan_cache_on_miss
        self.nusc_root = Path(nusc_root) if nusc_root else None
        self.nusc_version = nusc_version
        self.num_sweeps = num_sweeps
        self.seed = seed

        if self.scene_cache and not self.scene_cache.exists():
            raise FileNotFoundError(f"Scene cache path not found: {self.scene_cache}")
        if self.nusc_root and not self.nusc_root.exists():
            raise FileNotFoundError(f"nuScenes root not found: {self.nusc_root}")

        if self.nusc_root is None and self.scene_cache and self._looks_like_nuscenes_root(self.scene_cache):
            self.nusc_root = self.scene_cache

        self._nusc = None
        self._cache_index: Dict[str, Path] = {}
        self._cache_index_ready = False

    @staticmethod
    def _looks_like_nuscenes_root(path: Path) -> bool:
        return (path / "samples" / "LIDAR_TOP").exists() or (path / "sweeps" / "LIDAR_TOP").exists()

    def _ensure_nusc(self):
        if self.nusc_root is None:
            return None
        if NuScenes is None:
            raise ImportError(
                "nuscenes-devkit is required for raw nuScenes loading. Install it or provide a cache directory."
            )
        if self._nusc is None:
            self._nusc = NuScenes(version=self.nusc_version, dataroot=str(self.nusc_root), verbose=False)
        return self._nusc

    def get_display_scene_id(self, record: Dict[str, Any]) -> str:
        for key in ("scene_id", "scene", "scene_token"):
            value = record.get(key)
            if value:
                return str(value)
        meta = record.get("meta")
        if isinstance(meta, dict):
            for key in ("scene_id", "scene", "scene_token"):
                value = meta.get(key)
                if value:
                    return str(value)
        return extract_sample_token(record)

    def _candidate_cache_keys(self, record: Dict[str, Any], scene_id: str, sample_token: str) -> List[str]:
        keys = []
        for value in (
            scene_id,
            sample_token,
            record.get("scene_token"),
            record.get("token"),
            _get_nested(record, "meta", "scene_token"),
            _get_nested(record, "meta", "scene_id"),
        ):
            if value:
                keys.append(str(value))
        if self.nusc_root is not None:
            nusc = self._ensure_nusc()
            if nusc is not None and sample_token:
                try:
                    sample = nusc.get("sample", sample_token)
                    scene = nusc.get("scene", sample["scene_token"])
                except Exception:
                    scene = None
                if scene is not None:
                    for value in (scene.get("name"), scene.get("token")):
                        if value:
                            keys.append(str(value))

        deduped = []
        seen = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    def _build_cache_index(self):
        if self._cache_index_ready or self.scene_cache is None:
            return
        for file_path in self.scene_cache.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_CACHE_EXTENSIONS:
                self._cache_index.setdefault(file_path.stem, file_path)
                self._cache_index.setdefault(file_path.name, file_path)
        self._cache_index_ready = True

    def _find_cache_file(self, keys: Sequence[str]) -> Optional[Path]:
        if self.scene_cache is None or self._looks_like_nuscenes_root(self.scene_cache):
            return None

        candidate_paths: List[Path] = []
        for key in keys:
            for ext in SUPPORTED_CACHE_EXTENSIONS:
                candidate_paths.append(self.scene_cache / f"{key}{ext}")
                candidate_paths.append(self.scene_cache / key / f"scene{ext}")
                candidate_paths.append(self.scene_cache / key / f"points{ext}")
                candidate_paths.append(self.scene_cache / key / f"point_cloud{ext}")
                candidate_paths.append(self.scene_cache / key / f"{key}{ext}")

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        if self.scan_cache_on_miss:
            self._build_cache_index()
            for key in keys:
                if key in self._cache_index:
                    return self._cache_index[key]
        return None

    def _default_feat(self, num_points: int) -> np.ndarray:
        if self.feat_mode == "const_0p4":
            return np.full((num_points, 3), 0.4, dtype=np.float32)
        raise ValueError(f"Unsupported feat_mode: {self.feat_mode}")

    def _split_point_array(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(f"Expected NxC point array with C>=3, got shape {array.shape}")

        xyz = array[:, :3].astype(np.float32)
        if array.shape[1] >= 6:
            feat = array[:, 3:6].astype(np.float32)
        elif array.shape[1] in (4, 5):
            feat = np.repeat(array[:, 3:4].astype(np.float32), 3, axis=1)
        else:
            feat = self._default_feat(xyz.shape[0])
        return xyz, feat

    def _normalize_feat(self, feat: Optional[np.ndarray], num_points: int) -> np.ndarray:
        if feat is None:
            return self._default_feat(num_points)
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat[:, None]
        if feat.shape[0] != num_points:
            raise ValueError("Feature length does not match xyz length.")
        if feat.shape[1] == 1:
            feat = np.repeat(feat, 3, axis=1)
        elif feat.shape[1] == 2:
            feat = np.concatenate([feat, feat[:, :1]], axis=1)
        elif feat.shape[1] > 3:
            feat = feat[:, :3]
        return feat.astype(np.float32)

    def _load_cache_payload(self, cache_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if cache_path.suffix.lower() == ".npz":
            with np.load(cache_path, allow_pickle=True) as payload:
                data = {key: payload[key] for key in payload.files}
        elif cache_path.suffix.lower() == ".npy":
            data = np.load(cache_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
                data = data.item()
        else:
            data = torch.load(cache_path, map_location="cpu")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if isinstance(data, np.ndarray):
            return self._split_point_array(data)

        if isinstance(data, dict):
            xyz = None
            feat = None

            for key in ("xyz", "points_xyz", "coords"):
                value = data.get(key)
                if value is not None:
                    xyz = value
                    break

            for key in ("feat", "features", "colors", "rgb"):
                value = data.get(key)
                if value is not None:
                    feat = value
                    break

            if xyz is None:
                for key in ("pc", "points", "point_cloud", "scene"):
                    value = data.get(key)
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        return self._split_point_array(np.asarray(value))
                raise ValueError(f"Unsupported cache payload keys in {cache_path}")

            if isinstance(xyz, torch.Tensor):
                xyz = xyz.cpu().numpy()
            xyz = np.asarray(xyz, dtype=np.float32)

            if feat is not None and isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            feat = self._normalize_feat(feat, xyz.shape[0])
            return xyz, feat

        raise ValueError(f"Unsupported cache payload type {type(data)} from {cache_path}")

    def _sample_points(self, xyz: np.ndarray, feat: np.ndarray, sample_key: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.pointnum <= 0:
            return xyz, feat

        rng = np.random.default_rng(_stable_seed(sample_key, self.seed))
        replace = xyz.shape[0] < self.pointnum
        indices = rng.choice(xyz.shape[0], size=self.pointnum, replace=replace)
        return xyz[indices], feat[indices]

    @staticmethod
    def _pc_norm(xyz: np.ndarray) -> np.ndarray:
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        scale = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        if scale > 0:
            xyz = xyz / scale
        return xyz

    def _load_from_nuscenes(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        nusc = self._ensure_nusc()
        if nusc is None:
            raise FileNotFoundError("Scene cache miss and raw nuScenes root not configured.")

        sample = nusc.get("sample", sample_token)
        if self.num_sweeps > 1:
            if LidarPointCloud is None:
                raise ImportError(
                    "nuscenes-devkit with LidarPointCloud is required for multi-sweep loading."
                )
            points, _ = LidarPointCloud.from_file_multisweep(
                nusc,
                sample,
                chan="LIDAR_TOP",
                ref_chan="LIDAR_TOP",
                nsweeps=self.num_sweeps,
            )
            point_array = points.points.T.astype(np.float32)
            xyz = point_array[:, :3]
            feat = self._default_feat(xyz.shape[0])
            return xyz, feat

        lidar_token = sample["data"]["LIDAR_TOP"]
        sample_data = nusc.get("sample_data", lidar_token)
        lidar_path = Path(nusc.dataroot) / sample_data["filename"]
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size % 5 != 0:
            raise ValueError(f"Unexpected lidar bin format: {lidar_path}")
        point_array = raw.reshape(-1, 5)
        xyz = point_array[:, :3].astype(np.float32)
        feat = self._default_feat(xyz.shape[0])
        return xyz, feat

    def load_scene(self, record: Dict[str, Any], scene_id: str, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        keys = self._candidate_cache_keys(record, scene_id, sample_token)
        cache_path = self._find_cache_file(keys)
        if cache_path is not None:
            xyz, feat = self._load_cache_payload(cache_path)
        else:
            xyz, feat = self._load_from_nuscenes(sample_token)

        xyz, feat = self._sample_points(xyz, feat, sample_token or scene_id)
        if self.normalize_pc:
            xyz = self._pc_norm(xyz)
        return xyz.astype(np.float32), feat.astype(np.float32)


class CaptionDataset(Dataset):
    def __init__(self, records: Sequence[CaptionSceneRecord], raw_records: Dict[str, Dict[str, Any]], loader: SceneCacheLoader):
        self.records = list(records)
        self.raw_records = raw_records
        self.loader = loader

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        raw_record = self.raw_records[item.scene_id]
        xyz, feat = self.loader.load_scene(raw_record, item.scene_id, item.sample_token)
        return {
            "xyz": torch.from_numpy(xyz),
            "feat": torch.from_numpy(feat),
            "scene_id": item.scene_id,
            "sample_token": item.sample_token,
            "prompt": item.prompt,
            "ground_truth": item.ground_truth,
            "references": item.references,
        }


class QADataset(Dataset):
    def __init__(self, records: Sequence[QARecord], raw_records: Dict[str, Dict[str, Any]], loader: SceneCacheLoader):
        self.records = list(records)
        self.raw_records = raw_records
        self.loader = loader

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        raw_record = self.raw_records[item.question_id]
        xyz, feat = self.loader.load_scene(raw_record, item.scene_id, item.sample_token)
        return {
            "xyz": torch.from_numpy(xyz),
            "feat": torch.from_numpy(feat),
            "scene_id": item.scene_id,
            "sample_token": item.sample_token,
            "question_id": item.question_id,
            "question": item.question,
            "prompt": item.prompt,
            "ground_truth": item.ground_truth,
            "references": item.references,
            "question_type": item.question_type,
            "answer_type": item.answer_type,
        }


def caption_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "xyz": torch.stack([item["xyz"] for item in batch], dim=0),
        "feat": torch.stack([item["feat"] for item in batch], dim=0),
        "scene_id": [item["scene_id"] for item in batch],
        "sample_token": [item["sample_token"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "ground_truth": [item["ground_truth"] for item in batch],
        "references": [item["references"] for item in batch],
    }


def qa_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "xyz": torch.stack([item["xyz"] for item in batch], dim=0),
        "feat": torch.stack([item["feat"] for item in batch], dim=0),
        "scene_id": [item["scene_id"] for item in batch],
        "sample_token": [item["sample_token"] for item in batch],
        "question_id": [item["question_id"] for item in batch],
        "question": [item["question"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
        "ground_truth": [item["ground_truth"] for item in batch],
        "references": [item["references"] for item in batch],
        "question_type": [item["question_type"] for item in batch],
        "answer_type": [item["answer_type"] for item in batch],
    }


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, collate_fn):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def infer_question_type(record: Dict[str, Any], question: str) -> str:
    raw_type = (
        record.get("question_type")
        or record.get("template_type")
        or _get_nested(record, "meta", "question_type")
        or _get_nested(record, "meta", "template_type")
    )
    if raw_type:
        raw_type = str(raw_type).strip().lower()
        mapping = {
            "exist": "object_presence",
            "existence": "object_presence",
            "presence": "object_presence",
            "object_presence": "object_presence",
            "count": "counting",
            "counting": "counting",
            "relative_position": "relative_position",
            "position": "relative_position",
            "location": "relative_position",
            "relation": "relative_position",
            "attribute": "attribute",
            "color": "attribute",
            "size": "attribute",
            "type": "attribute",
            "motion": "motion_state",
            "moving": "motion_state",
            "motion_state": "motion_state",
            "map": "map_context",
            "road": "map_context",
            "lane": "map_context",
            "map_context": "map_context",
        }
        if raw_type in mapping:
            return mapping[raw_type]

    lowered = question.lower()
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", lowered):
        return "counting"
    if re.search(r"\bis there\b|\bare there\b|\bany\b.*\bvisible\b|\bdo you see\b", lowered):
        return "object_presence"
    if re.search(r"\bleft of\b|\bright of\b|\bin front of\b|\bbehind\b|\bnext to\b|\bclosest\b", lowered):
        return "relative_position"
    if re.search(r"\bmoving\b|\bstopped\b|\bparked\b|\bspeed\b|\bmotion\b", lowered):
        return "motion_state"
    if re.search(r"\blane\b|\broad\b|\bintersection\b|\bcrosswalk\b|\btraffic light\b|\bmap\b", lowered):
        return "map_context"
    return "attribute"


def infer_answer_type(answer: str, record: Optional[Dict[str, Any]] = None) -> str:
    raw_type = None
    if isinstance(record, dict):
        raw_type = record.get("answer_type") or _get_nested(record, "meta", "answer_type")
    if raw_type:
        raw_type = str(raw_type).strip().lower()
        if raw_type in {"yes/no", "yesno"}:
            return "yes/no"
        if raw_type in {"number", "count"}:
            return "number"
        if raw_type in {"category", "label"}:
            return "category"
        if raw_type in {"phrase", "free-form", "free_form"}:
            return "free-form phrase"

    lowered = clean_text(answer).lower()
    if lowered in {"yes", "no", "true", "false"}:
        return "yes/no"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", lowered):
        return "number"
    if len(lowered.split()) <= 3:
        return "category"
    return "free-form phrase"


def load_caption_records(
    annotation_path: str,
    loader: SceneCacheLoader,
    prompt_template: str = CAPTION_PROMPT,
) -> Tuple[List[CaptionSceneRecord], Dict[str, Dict[str, Any]]]:
    records = load_json_records(annotation_path)
    grouped: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    for record in records:
        ground_truth = extract_caption_ground_truth(record)
        if not ground_truth:
            continue
        scene_id = loader.get_display_scene_id(record)
        sample_token = extract_sample_token(record)
        slot = grouped.setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "sample_token": sample_token,
                "prompt": prompt_template,
                "ground_truth": ground_truth,
                "references": [],
                "record": record,
            },
        )
        if ground_truth not in slot["references"]:
            slot["references"].append(ground_truth)

    output_records = []
    raw_records: Dict[str, Dict[str, Any]] = {}
    for scene_id, payload in grouped.items():
        output_records.append(
            CaptionSceneRecord(
                scene_id=scene_id,
                sample_token=payload["sample_token"],
                prompt=payload["prompt"],
                ground_truth=payload["ground_truth"],
                references=list(payload["references"]),
            )
        )
        raw_records[scene_id] = payload["record"]
    return output_records, raw_records


def load_qa_records(
    annotation_path: str,
    loader: SceneCacheLoader,
    prompt_template: str = QA_PROMPT_TEMPLATE,
) -> Tuple[List[QARecord], Dict[str, Dict[str, Any]]]:
    records = load_json_records(annotation_path)
    output_records: List[QARecord] = []
    raw_records: Dict[str, Dict[str, Any]] = {}

    for index, record in enumerate(records):
        question = extract_question(record)
        ground_truth = extract_qa_ground_truth(record)
        if not question or not ground_truth:
            continue

        scene_id = loader.get_display_scene_id(record)
        sample_token = extract_sample_token(record)
        question_id = (
            record.get("question_id")
            or record.get("qid")
            or record.get("id")
            or _get_nested(record, "meta", "question_id")
            or f"{scene_id}_{index:06d}"
        )
        question_type = infer_question_type(record, question)
        answer_type = infer_answer_type(ground_truth, record)

        output_records.append(
            QARecord(
                scene_id=scene_id,
                sample_token=sample_token,
                question_id=str(question_id),
                question=question,
                prompt=prompt_template.format(question=question),
                ground_truth=ground_truth,
                question_type=question_type,
                answer_type=answer_type,
                references=[ground_truth],
            )
        )
        raw_records[str(question_id)] = record

    return output_records, raw_records


def _move_tensor_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = dict(batch)
    out["xyz"] = batch["xyz"].to(device, non_blocking=True)
    out["feat"] = batch["feat"].to(device, non_blocking=True)
    return out


@torch.inference_mode()
def generate_caption_predictions(
    model,
    dataloader,
    device: str,
    max_new_tokens: int,
    num_beams: int = 1,
    min_length: int = 1,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> List[Dict[str, Any]]:
    predictions = []
    for batch in tqdm(dataloader, desc="Caption inference"):
        batch = _move_tensor_batch(batch, device)
        prompts = build_generation_prompts(batch["prompt"])
        answers = model.generate(
            texts=prompts,
            xyz=batch["xyz"],
            feat=batch["feat"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        for scene_id, sample_token, ground_truth, references, answer in zip(
            batch["scene_id"],
            batch["sample_token"],
            batch["ground_truth"],
            batch["references"],
            answers,
        ):
            predictions.append(
                {
                    "scene_id": scene_id,
                    "sample_token": sample_token,
                    "prediction": postprocess_generation(answer),
                    "ground_truth": ground_truth,
                    "references": list(references),
                }
            )
    return predictions


@torch.inference_mode()
def generate_qa_predictions(
    model,
    dataloader,
    device: str,
    max_new_tokens: int,
    num_beams: int = 1,
    min_length: int = 1,
    temperature: float = 0.0,
    do_sample: bool = False,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
) -> List[Dict[str, Any]]:
    predictions = []
    for batch in tqdm(dataloader, desc="QA inference"):
        batch = _move_tensor_batch(batch, device)
        prompts = build_generation_prompts(batch["prompt"])
        answers = model.generate(
            texts=prompts,
            xyz=batch["xyz"],
            feat=batch["feat"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        for scene_id, sample_token, question_id, question, ground_truth, references, question_type, answer_type, answer in zip(
            batch["scene_id"],
            batch["sample_token"],
            batch["question_id"],
            batch["question"],
            batch["ground_truth"],
            batch["references"],
            batch["question_type"],
            batch["answer_type"],
            answers,
        ):
            predictions.append(
                {
                    "scene_id": scene_id,
                    "sample_token": sample_token,
                    "question_id": question_id,
                    "question": question,
                    "prediction": postprocess_generation(answer),
                    "ground_truth": ground_truth,
                    "references": list(references),
                    "question_type": question_type,
                    "answer_type": answer_type,
                }
            )
    return predictions


def save_json(path: str, payload: Any):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
