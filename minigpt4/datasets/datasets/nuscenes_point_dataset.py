import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class NuScenesPointCloudDataset(Dataset):
    """
    NuScenes dataset aligned to the same sample dict contract as ObjectPointCloudDataset.
    - Reads JSON (list) or JSONL annotations containing `sample_token` and `conversations`.
    - Loads LiDAR point cloud via nuScenes devkit and sample_token.
    - Returns:
        single-turn: pc, instruction_input, answer, PC_id
        multi-turn:  pc, conv_q, conv_a, connect_sym, PC_id
    """
    def __init__(
        self,
        text_processor,
        nusc,                          # NuScenes devkit instance, built by builder
        anno_path: str,
        pointnum: int = 8192,
        normalize_pc: bool = True,
        use_intensity: bool = True,
        use_time_or_ring: bool = True,
        return_dim: int = 6,           # keep 6 to match existing model assumptions
    ):
        self.text_processor = text_processor
        self.nusc = nusc
        self.anno_path = anno_path
        self.pointnum = pointnum
        self.normalize_pc = normalize_pc

        self.use_intensity = use_intensity
        self.use_time_or_ring = use_time_or_ring
        self.return_dim = return_dim

        self.point_indicator = "<point>"
        self.connect_sym = "!@#"

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
            self.list_data_dict = items
        else:
            with open(anno_path, "r", encoding="utf-8") as f:
                self.list_data_dict = json.load(f)

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

    def _build_pc_features(self, pts5: np.ndarray) -> np.ndarray:
        xyz = pts5[:, 0:3]
        intensity = pts5[:, 3:4] if self.use_intensity else np.zeros((pts5.shape[0], 1), np.float32)
        extra = pts5[:, 4:5] if self.use_time_or_ring else np.zeros((pts5.shape[0], 1), np.float32)

        # Make Nx6 by default: [x,y,z,intensity,extra,0]
        pc6 = np.concatenate([xyz, intensity, extra, np.zeros_like(extra)], axis=1).astype(np.float32)

        if self.return_dim == 6:
            return pc6
        elif self.return_dim < 6:
            return pc6[:, : self.return_dim]
        else:
            # pad if someone wants >6
            pad = np.zeros((pc6.shape[0], self.return_dim - 6), dtype=np.float32)
            return np.concatenate([pc6, pad], axis=1)

    def _sample_points(self, pc: np.ndarray) -> np.ndarray:
        N = pc.shape[0]
        if N >= self.pointnum:
            idx = np.random.choice(N, self.pointnum, replace=False)
        else:
            idx = np.random.choice(N, self.pointnum, replace=True)
        return pc[idx]

    def _pc_norm(self, pc: np.ndarray) -> np.ndarray:
        xyz = pc[:, :3]
        other = pc[:, 3:]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if m > 0:
            xyz = xyz / m
        return np.concatenate([xyz, other], axis=1)

    # ===== main getitem =====
    def __getitem__(self, index: int):
        sources = self.list_data_dict[index]
        if isinstance(index, int):
            sources = [sources]
        assert len(sources) == 1, "sources should be a list"
        sources = sources[0]

        sample_token = sources["sample_token"]

        # load point cloud only if prompt contains <point> like objaverse
        # (keeps behavior consistent)
        if self.point_indicator in sources["conversations"][0]["value"]:
            lidar_path = self._get_lidar_path(sample_token)
            pts5 = self._read_lidar_bin(lidar_path)
            pc = self._build_pc_features(pts5)
            pc = self._sample_points(pc)
            if self.normalize_pc:
                pc = self._pc_norm(pc)
            point_cloud = torch.from_numpy(pc.astype(np.float32))
        else:
            # If ever needed, but most records should include <point>
            point_cloud = None

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
                "pc": point_cloud,
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
            "pc": point_cloud,
            "instruction_input": instruction,
            "answer": answer,
            "PC_id": sample_token,
        }