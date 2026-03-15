import os
import json

from nuscenes.nuscenes import NuScenes

# Import dataset class bạn đã tạo
from minigpt4.datasets.datasets.nuscenes_point_dataset import NuScenesPointCloudDataset

# Nếu text_processor của repo bạn phức tạp, tạm dùng identity để smoke test
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

def main():
    # ====== EDIT THESE PATHS ======
    NUSC_ROOT = "../"                # nuScenes root
    VERSION = "v1.0-trainval"                  
    ANN_JSONL = "ann/nucaption_train.jsonl"    

    # ====== quick check annotation file ======
    rows = peek_jsonl(ANN_JSONL, n=2)
    print(f"Loaded {len(rows)} rows from jsonl. First keys:", rows[0].keys())
    print("First sample_token:", rows[0]["sample_token"])
    print("First human prompt (raw):", rows[0]["conversations"][0]["value"][:120], "...")
    print("First answer (raw):", rows[0]["conversations"][1]["value"][:120], "...")
    print("-" * 80)

    # ====== init nuScenes devkit ======
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)

    # ====== build dataset ======
    ds = NuScenesPointCloudDataset(
        text_processor=IdentityTextProcessor(),  # thay bằng processor thật nếu muốn
        nusc=nusc,
        anno_path=ANN_JSONL,
        pointnum=8192,
        normalize_pc=True,
        use_intensity=True,
        use_time_or_ring=True,
        return_dim=6,
    )

    # ====== fetch one sample ======
    sample = ds[0]
    print("Returned keys:", list(sample.keys()))
    pc = sample["pc"]
    print("pc dtype:", pc.dtype, "pc shape:", tuple(pc.shape))
    print("PC_id:", sample["PC_id"])

    # contract check (Objaverse single-turn)
    assert "instruction_input" in sample and "answer" in sample, "Not single-turn format"
    assert pc.ndim == 2 and pc.shape[0] == 8192, "Unexpected point shape"
    assert pc.shape[1] in (3, 4, 5, 6), "Unexpected point feature dim"

    print("\n--- instruction_input (first 200 chars) ---")
    print(sample["instruction_input"][:200])

    print("\n--- answer (first 200 chars) ---")
    print(sample["answer"][:200])

    print("\n✅ Dataset smoke test passed for ds[0].")

if __name__ == "__main__":
    main()