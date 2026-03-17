import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

class IdentityTextProcessor:
    def __call__(self, x: str) -> str:
        return x


def peek_jsonl(path, n=2):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def validate_contract(rows):
    if not rows:
        raise ValueError("Annotation file is empty.")

    for i, row in enumerate(rows, 1):
        if "sample_token" not in row:
            raise ValueError(f"Row {i} missing 'sample_token'.")

        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"Row {i} has invalid 'conversations'.")

        first = conversations[0]
        second = conversations[1]
        if first.get("from") != "human":
            raise ValueError(f"Row {i} first conversation role must be 'human'.")
        if second.get("from") != "assistant":
            raise ValueError(f"Row {i} second conversation role must be 'assistant'.")
        if "<point>" not in first.get("value", ""):
            raise ValueError(f"Row {i} human prompt is missing '<point>'.")


def detect_nusc_root(cli_root):
    candidates = []
    if cli_root:
        candidates.append(Path(cli_root))

    env_root = os.environ.get("NUSC_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            Path("."),
            Path(".."),
            Path("../data"),
            Path("../../data"),
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "samples" / "LIDAR_TOP").exists() or (candidate / "sweeps" / "LIDAR_TOP").exists():
            return candidate

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="NuScenes dataset smoke test")
    parser.add_argument("--ann-jsonl", default="ann/nucaption_train.jsonl", help="Path to JSONL annotation file.")
    parser.add_argument("--nusc-root", default=None, help="nuScenes root directory. Falls back to $NUSC_ROOT and common local paths.")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes dataset version.")
    parser.add_argument("--pointnum", type=int, default=8192, help="Number of points to sample.")
    parser.add_argument("--return-dim", type=int, default=6, help="Expected total point dimension across xyz+feat.")
    parser.add_argument("--feat-mode", default="const_0p4", help="Feature construction mode for NuScenes dataset.")
    parser.add_argument("--view", action="store_true", help="Open an Open3D viewer for the sampled point cloud.")
    return parser.parse_args()


def load_nuscenes_dataset_class():
    dataset_path = Path(__file__).resolve().parent / "minigpt4" / "datasets" / "datasets" / "nuscenes_point_dataset.py"
    spec = importlib.util.spec_from_file_location("nuscenes_point_dataset_smoke", dataset_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dataset module from {dataset_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NuScenesPointCloudDataset


def split_point_sample(sample):
    if "xyz" in sample and "feat" in sample:
        xyz = sample["xyz"]
        feat = sample["feat"]
        return xyz, feat

    if "pc" in sample:
        pc = sample["pc"]
        xyz = pc[:, :3]
        feat = pc[:, 3:] if pc.shape[1] > 3 else None
        return xyz, feat

    raise KeyError("Sample must contain either `xyz`/`feat` or `pc`.")


def view_point_cloud(xyz_tensor, feat_tensor, sample_id):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Missing 'open3d'. Install it in the current environment or run without --view."
        ) from exc

    xyz = xyz_tensor.cpu().numpy()
    feat = feat_tensor.cpu().numpy() if feat_tensor is not None else None

    if feat is not None and feat.shape[1] >= 1:
        intensity = feat[:, 0]
        lo = float(intensity.min())
        hi = float(intensity.max())
        if hi > lo:
            scaled = (intensity - lo) / (hi - lo)
        else:
            scaled = np.full_like(intensity, 0.5, dtype=np.float32)
        colors = np.repeat(scaled[:, None], 3, axis=1)
    else:
        colors = np.full((xyz.shape[0], 3), 0.7, dtype=np.float32)

    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    geom.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    print(f"Opening Open3D viewer for sample {sample_id}...")
    o3d.visualization.draw_geometries([geom], window_name=f"NuScenes Smoke Test: {sample_id}")


def main():
    args = parse_args()

    ann_path = Path(args.ann_jsonl)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    rows = peek_jsonl(ann_path, n=2)
    validate_contract(rows)

    print(f"Loaded {len(rows)} rows from jsonl. First keys:", rows[0].keys())
    print("First sample_token:", rows[0]["sample_token"])
    print("First human prompt (raw):", rows[0]["conversations"][0]["value"][:120], "...")
    print("First answer (raw):", rows[0]["conversations"][1]["value"][:120], "...")
    print("-" * 80)

    nusc_root = detect_nusc_root(args.nusc_root)
    if nusc_root is None:
        raise FileNotFoundError(
            "Could not detect a nuScenes root. Pass --nusc-root or set NUSC_ROOT "
            "to a directory containing samples/LIDAR_TOP or sweeps/LIDAR_TOP."
        )

    print("Using nuScenes root:", nusc_root)

    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise ImportError(
            "Missing 'nuscenes-devkit'. Install the project environment before running the smoke test."
        ) from exc

    try:
        NuScenesPointCloudDataset = load_nuscenes_dataset_class()
    except ImportError as exc:
        raise ImportError(
            "Failed to import NuScenesPointCloudDataset. This usually means the current Python "
            "environment is missing project dependencies such as torch."
        ) from exc

    nusc = NuScenes(version=args.version, dataroot=str(nusc_root), verbose=False)

    ds = NuScenesPointCloudDataset(
        text_processor=IdentityTextProcessor(),
        nusc=nusc,
        anno_path=str(ann_path),
        pointnum=args.pointnum,
        normalize_pc=True,
        feat_mode=args.feat_mode,
    )

    sample = ds[0]
    print("Returned keys:", list(sample.keys()))
    xyz, feat = split_point_sample(sample)
    total_dim = xyz.shape[1] + (0 if feat is None else feat.shape[1])
    print("xyz dtype:", xyz.dtype, "xyz shape:", tuple(xyz.shape))
    if feat is not None:
        print("feat dtype:", feat.dtype, "feat shape:", tuple(feat.shape))
    else:
        print("feat: None")
    print("total point dim:", total_dim)
    print("PC_id:", sample["PC_id"])

    assert "instruction_input" in sample and "answer" in sample, "Not single-turn format"
    assert xyz.ndim == 2 and xyz.shape == (args.pointnum, 3), "Unexpected xyz shape"
    if feat is not None:
        assert feat.ndim == 2 and feat.shape[0] == args.pointnum, "Unexpected feat shape"
    assert total_dim == args.return_dim, "Unexpected total point feature dim"

    print("\n--- instruction_input (first 200 chars) ---")
    print(sample["instruction_input"][:200])

    print("\n--- answer (first 200 chars) ---")
    print(sample["answer"][:200])

    if args.view:
        view_point_cloud(xyz, feat, sample["PC_id"])

    print("\n✅ Dataset smoke test passed for ds[0].")

if __name__ == "__main__":
    main()
