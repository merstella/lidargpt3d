# tools/dump_nusc_to_jsonl.py
# Dump nuCaption + nuScenesQA into internal JSONL format (single-turn, keep all records)
# - nuCaption: use answer_lidar
# - keep prompts as-is (no sanitize)
# - output format per line:
#   {"sample_token": "...", "conversations":[{"from":"human","value":"<point>\n...q..."},{"from":"assistant","value":"...a..."}]}

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_nucaption(in_path: str, out_path: str) -> None:
    """
    Input: nuCaption json (a list of dicts)
    Uses: question + answer_lidar (required)
    Keeps: all records (no dedup)
    """
    data = json.load(open(in_path, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list for nuCaption, got: {type(data)}")

    out_records = []
    missing = 0

    for r in data:
        sample_token = r.get("sample_token")
        q = (r.get("question") or "").strip()
        a = (r.get("answer_lidar") or "").strip()

        if not sample_token or not q or not a:
            missing += 1
            continue

        out_records.append(
            {
                "sample_token": sample_token,
                "conversations": [
                    {"from": "human", "value": f"<point>\n{q}"},
                    {"from": "assistant", "value": a},
                ],
            }
        )

    _write_jsonl(out_records, out_path)
    print(
        f"[nuCaption] {in_path} -> {out_path} | kept={len(out_records)} | skipped_missing={missing}"
    )


def dump_nuscenesqa(in_path: str, out_path: str) -> None:
    """
    Input: nuScenesQA json (dict with 'info' and 'questions' list)
    Uses: question + answer
    Keeps: all records (no dedup)
    """
    obj = json.load(open(in_path, "r", encoding="utf-8"))
    questions = obj.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Expected nuScenesQA JSON with key 'questions' as a list.")

    out_records = []
    missing = 0

    for r in questions:
        sample_token = r.get("sample_token")
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "")
        a = str(a).strip()

        if not sample_token or not q or not a:
            missing += 1
            continue

        out_records.append(
            {
                "sample_token": sample_token,
                "conversations": [
                    {"from": "human", "value": f"<point>\n{q}"},
                    {"from": "assistant", "value": a},
                ],
                # optional metadata (doesn't affect training)
                "meta": {
                    "template_type": r.get("template_type"),
                    "num_hop": r.get("num_hop"),
                },
            }
        )

    _write_jsonl(out_records, out_path)
    print(
        f"[nuScenesQA] {in_path} -> {out_path} | kept={len(out_records)} | skipped_missing={missing}"
    )


if __name__ == "__main__":
    # ---- EDIT PATHS HERE ----
    # nuCaption
    dump_nucaption("data/nucaption_train.json", "ann/nucaption_train.jsonl")
    dump_nucaption("data/nucaption_val.json", "ann/nucaption_val.jsonl")

    # nuScenesQA
    dump_nuscenesqa("data/nuscenesqa_train.json", "ann/nuscenesqa_train.jsonl")
    dump_nuscenesqa("data/nuscenesqa_val.json", "ann/nuscenesqa_val.jsonl")