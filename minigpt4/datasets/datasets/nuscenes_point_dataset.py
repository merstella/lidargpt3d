import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

class NuScenesPointCloudDataset(Dataset):
    """
    NuScenes dataset aligned to the same text contract as ObjectPointCloudDataset.
    Point data follows Uni3D-style interface: separated xyz and feat.
    - Reads JSON (list) or JSONL annotations containing `sample_token` and `conversations`.
    - Loads LiDAR point cloud via nuScenes devkit and sample_token.
    - Returns:
        single-turn: xyz, feat, instruction_input, answer, PC_id
        multi-turn:  xyz, feat, conv_q, conv_a, connect_sym, PC_id
    """
    def __init__(
        self,
        text_processor,
        nusc,                          # NuScenes devkit instance, built by builder
        anno_path: str,
        pointnum: int = 8192,
        normalize_pc: bool = False,
        use_intensity: bool = True,
        use_time_or_ring: bool = True,
        return_dim: int = 6,           # keep 6 to match existing model assumptions
        feat_mode: str = "const_0p4",
    ):
        self.text_processor = text_processor
        self.nusc = nusc
        self.anno_path = anno_path
        self.pointnum = pointnum
        self.normalize_pc = normalize_pc

        self.use_intensity = use_intensity
        self.use_time_or_ring = use_time_or_ring
        self.return_dim = return_dim
        self.feat_mode = feat_mode

        self.point_indicator = "<point>"
        self.connect_sym = "!@#"

        self._validate_config()

        # --- load annotations (support json list or jsonl) ---
        print(f"Loading anno file from {anno_path}.")
        if anno_path.endswith(".jsonl"):
            items = []
            with open(anno_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
            loaded_data = items
        else:
            with open(anno_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)

        self.list_data_dict = self._filter_valid_records(loaded_data)

        print(f"Dataset size: {len(self.list_data_dict)}.")

        # --- cache sample_token -> lidar filepath to speed up ---
        self._lidar_path_cache = {}

    def __len__(self):
        return len(self.list_data_dict)

    # ===== point cloud utils =====
    def _get_lidar_path(self, sample_token: str) -> str:
        if sample_token in self._lidar_path_cache:
            return self._lidar_path_cache[sample_token]

        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, sd["filename"])
        self._lidar_path_cache[sample_token] = lidar_path
        return lidar_path

    def _read_lidar_bin(self, lidar_path: str) -> np.ndarray:
        raw = np.fromfile(lidar_path, dtype=np.float32)
        if raw.size % 5 != 0:
            raise ValueError(f"Unexpected lidar bin format: size%5!=0 for {lidar_path}")
        pts = raw.reshape(-1, 5)  # x,y,z,intensity,time_or_ring
        return pts

    def _build_xyz_feat(self, pts5: np.ndarray):
        xyz = pts5[:, 0:3].astype(np.float32)

        if self.feat_mode == "const_0p4":
            feat = np.full((xyz.shape[0], 3), 0.4, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported feat_mode: {self.feat_mode}")

        return xyz, feat

    def _sample_points(self, xyz: np.ndarray, feat: np.ndarray):
        point_count = xyz.shape[0]
        if point_count >= self.pointnum:
            idx = np.random.choice(point_count, self.pointnum, replace=False)
        else:
            idx = np.random.choice(point_count, self.pointnum, replace=True)
        return xyz[idx], feat[idx]

    def _pc_norm(self, xyz: np.ndarray) -> np.ndarray:
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if m > 0:
            xyz = xyz / m
        return xyz

    def _validate_config(self):
        supported_modes = {"const_0p4"}
        if self.feat_mode not in supported_modes:
            raise ValueError(
                f"Unsupported feat_mode `{self.feat_mode}`. Supported: {sorted(supported_modes)}."
            )

        if self.return_dim != 6:
            raise ValueError(
                f"return_dim={self.return_dim} is not supported in Uni3D-style mode. Use return_dim=6."
            )

    def _filter_valid_records(self, records):
        if isinstance(records, dict):
            if "annotations" in records and isinstance(records["annotations"], list):
                records = records["annotations"]
            else:
                raise ValueError("Expected annotation list or dict with `annotations` list.")
        if not isinstance(records, list):
            raise ValueError("Annotations must be a list.")

        filtered = []
        dropped_missing_token = 0
        dropped_invalid_conv = 0
        dropped_missing_point = 0

        for item in records:
            if not isinstance(item, dict):
                dropped_invalid_conv += 1
                continue

            sample_token = item.get("sample_token")
            if not sample_token:
                dropped_missing_token += 1
                continue

            conversations = item.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                dropped_invalid_conv += 1
                continue

            first_turn = conversations[0]
            first_value = first_turn.get("value", "") if isinstance(first_turn, dict) else ""
            if self.point_indicator not in first_value:
                dropped_missing_point += 1
                continue

            filtered.append(item)

        dropped_total = dropped_missing_token + dropped_invalid_conv + dropped_missing_point
        logging.info(
            "NuScenes filtering: kept=%d dropped=%d (missing_token=%d invalid_conv=%d missing_point=%d)",
            len(filtered),
            dropped_total,
            dropped_missing_token,
            dropped_invalid_conv,
            dropped_missing_point,
        )
        print(
            f"NuScenes filtering: kept={len(filtered)} dropped={dropped_total} "
            f"(missing_token={dropped_missing_token}, invalid_conv={dropped_invalid_conv}, "
            f"missing_point={dropped_missing_point})."
        )
        return filtered

    # ===== main getitem =====
    def __getitem__(self, index: int):
        sources = self.list_data_dict[index]

        sample_token = sources["sample_token"]

        lidar_path = self._get_lidar_path(sample_token)
        pts5 = self._read_lidar_bin(lidar_path)
        xyz, feat = self._build_xyz_feat(pts5)
        xyz, feat = self._sample_points(xyz, feat)
        if self.normalize_pc:
            xyz = self._pc_norm(xyz)
        point_xyz = torch.from_numpy(xyz.astype(np.float32))
        point_feat = torch.from_numpy(feat.astype(np.float32))

        # multi-turn: len(conversations) > 2
        if len(sources["conversations"]) > 2:
            first_instruction = sources["conversations"][0]["value"].replace("<point>", "").replace("\n", "").strip()
            first_instruction = "<PC><PointCloudHere></PC> {} ".format(first_instruction)

            questions = [first_instruction]
            answers = []
            for i, item in enumerate(sources["conversations"][1:]):
                if i % 2 == 0:   # assistant
                    answers.append(item["value"])
                else:            # human
                    questions.append(item["value"] + " ")

            return {
                "xyz": point_xyz,
                "feat": point_feat,
                "conv_q": self.connect_sym.join(questions),
                "conv_a": self.connect_sym.join(answers),
                "PC_id": sample_token,
                "connect_sym": self.connect_sym,
            }

        # single-turn
        instruction = sources["conversations"][0]["value"]
        instruction = instruction.replace("<point>", "").replace("\n", "").strip()
        instruction = "<PC><PointCloudHere></PC> {} ".format(self.text_processor(instruction))

        answer = sources["conversations"][1]["value"]
        return {
            "xyz": point_xyz,
            "feat": point_feat,
            "instruction_input": instruction,
            "answer": answer,
            "PC_id": sample_token,
        }
