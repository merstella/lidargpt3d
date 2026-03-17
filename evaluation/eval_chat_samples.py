import argparse
import os
import random
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics.common import (
    CaptionDataset,
    SceneCacheLoader,
    build_dataloader,
    build_generation_prompts,
    caption_collate,
    ensure_dir,
    init_model_for_eval,
    load_caption_records,
    postprocess_generation,
    save_json,
)


DEFAULT_PROMPTS = [
    "Describe the current driving scene.",
    "What road users are closest to the ego vehicle?",
    "Are there any immediate hazards the driver should watch?",
    "Describe the road layout and drivable space.",
    "What objects are moving in this scene?",
    "What map cues or infrastructure are visible?",
    "Summarize the scene in one short paragraph.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative multi-turn sampling for nuScenes scenes.")
    parser.add_argument("--cfg-path", default="./eval_configs/benchmark_evaluation_nuscenes.yaml")
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--scene-cache", required=True)
    parser.add_argument("--nusc-root", default=None)
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--second-checkpoint", default=None)
    parser.add_argument("--output-root", default="evaluation/outputs")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--stage-name", default="stage4")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--num-sweeps", type=int, default=1)
    parser.add_argument("--normalize-pc", action="store_true")
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--prompts-per-scene", type=int, default=7)
    parser.add_argument("--prompt-file", default=None, help="Optional txt/json prompt list")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--options", nargs="+", default=None)
    return parser.parse_args()


def load_prompt_bank(prompt_file: str):
    if prompt_file is None:
        return list(DEFAULT_PROMPTS)
    if prompt_file.endswith(".json"):
        import json

        with open(prompt_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    with open(prompt_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


@torch.inference_mode()
def answer_scene(model, xyz, feat, prompts, max_new_tokens, temperature, num_beams):
    model_prompts = build_generation_prompts(prompts)
    xyz = xyz.repeat(len(prompts), 1, 1)
    feat = feat.repeat(len(prompts), 1, 1)
    answers = model.generate(
        texts=model_prompts,
        xyz=xyz,
        feat=feat,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        num_beams=num_beams,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        min_length=1,
    )
    return [postprocess_generation(answer) for answer in answers]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    prompt_bank = load_prompt_bank(args.prompt_file)
    if not 20 <= args.num_scenes <= 50:
        raise ValueError("--num-scenes must be within 20-50.")
    if not 5 <= args.prompts_per_scene <= 10:
        raise ValueError("--prompts-per-scene must be within 5-10.")
    if len(prompt_bank) < args.prompts_per_scene:
        raise ValueError("Prompt bank has fewer prompts than prompts-per-scene.")

    scene_loader = SceneCacheLoader(
        scene_cache=args.scene_cache,
        pointnum=args.pointnum,
        normalize_pc=args.normalize_pc,
        nusc_root=args.nusc_root,
        nusc_version=args.nusc_version,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
    )
    caption_records, raw_records = load_caption_records(args.annotation_path, scene_loader)
    if len(caption_records) < args.num_scenes:
        raise ValueError("Not enough scenes available for requested qualitative sample count.")

    sampled_records = rng.sample(caption_records, args.num_scenes)
    dataset = CaptionDataset(sampled_records, raw_records, scene_loader)
    dataloader = build_dataloader(dataset, args.batch_size, args.num_workers, caption_collate)

    model, _ = init_model_for_eval(
        cfg_path=args.cfg_path,
        device=args.device,
        checkpoint=args.checkpoint,
        second_checkpoint=args.second_checkpoint,
        cfg_options=args.options,
    )

    output_dir = ensure_dir(os.path.join(args.output_root, args.experiment_name, args.stage_name, "chat"))
    output_path = os.path.join(output_dir, "chat_samples.json")
    results = []

    for batch in dataloader:
        xyz = batch["xyz"].to(args.device)
        feat = batch["feat"].to(args.device)
        for index, scene_id in enumerate(batch["scene_id"]):
            prompts = rng.sample(prompt_bank, args.prompts_per_scene)
            answers = answer_scene(
                model=model,
                xyz=xyz[index : index + 1],
                feat=feat[index : index + 1],
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_beams=args.num_beams,
            )
            results.append(
                {
                    "scene_id": scene_id,
                    "dialogue": [
                        {"question": prompt, "answer": answer}
                        for prompt, answer in zip(prompts, answers)
                    ],
                }
            )

    save_json(output_path, results)


if __name__ == "__main__":
    main()
