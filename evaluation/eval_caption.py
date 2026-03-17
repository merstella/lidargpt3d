import argparse
import gc
import os
import random
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics.common import (
    CAPTION_PROMPT,
    CaptionDataset,
    SceneCacheLoader,
    build_dataloader,
    build_run_specs,
    caption_collate,
    ensure_dir,
    generate_caption_predictions,
    init_model_for_eval,
    load_caption_records,
    parse_ablation_tags,
    save_json,
)
from evaluation.metrics.reporting import update_paper_report


def parse_args():
    parser = argparse.ArgumentParser(description="Caption evaluation for nu-Caption.")
    parser.add_argument("--cfg-path", default="./eval_configs/benchmark_evaluation_nuscenes.yaml")
    parser.add_argument("--annotation-path", required=True, help="nu-Caption val json/jsonl")
    parser.add_argument("--scene-cache", required=True, help="Scene cache directory or raw nuScenes root")
    parser.add_argument("--nusc-root", default=None, help="Optional raw nuScenes root for cache misses")
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--output-root", default="evaluation/outputs")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--stage-name", default="stage4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--second-checkpoint", default=None)
    parser.add_argument("--checkpoint-spec", nargs="*", default=None, help="Repeated stage=checkpoint")
    parser.add_argument("--second-checkpoint-spec", nargs="*", default=None, help="Repeated stage=checkpoint")
    parser.add_argument("--prompt-template", default=CAPTION_PROMPT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--num-sweeps", type=int, default=1)
    parser.add_argument("--normalize-pc", action="store_true")
    parser.add_argument("--scan-scene-cache", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metric-device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--hallucination-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-model", default="./params_weight/bert-base-uncased")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--ablation-tag", action="append", default=[])
    parser.add_argument("--options", nargs="+", default=None, help="Extra model cfg overrides")
    return parser.parse_args()


def build_hallucination_audit(predictions, sample_size: int, seed: int, output_path: str):
    sample_size = min(sample_size, len(predictions))
    sampled = random.Random(seed).sample(predictions, sample_size) if sample_size else []
    payload = [
        {
            "scene_id": item["scene_id"],
            "prediction": item["prediction"],
            "ground_truth": item["ground_truth"],
            "hallucination_flag": None,
        }
        for item in sampled
    ]
    save_json(output_path, payload)
    return {
        "sampled_scenes": len(payload),
        "flagged_count": None,
        "pending_count": len(payload),
        "path": output_path,
    }


def main():
    args = parse_args()
    from evaluation.metrics.text_metrics import evaluate_caption_predictions

    random.seed(args.seed)
    metric_device = args.metric_device or args.device
    if args.temperature != 0.0 or args.do_sample or args.num_beams != 1:
        raise ValueError(
            "Caption evaluation must use temperature=0, greedy decoding, and num_beams=1."
        )

    run_specs = build_run_specs(
        stage_name=args.stage_name,
        checkpoint=args.checkpoint,
        second_checkpoint=args.second_checkpoint,
        checkpoint_spec=args.checkpoint_spec,
        second_checkpoint_spec=args.second_checkpoint_spec,
    )
    ablation_tags = parse_ablation_tags(args.ablation_tag)

    scene_loader = SceneCacheLoader(
        scene_cache=args.scene_cache,
        pointnum=args.pointnum,
        normalize_pc=args.normalize_pc,
        scan_cache_on_miss=args.scan_scene_cache,
        nusc_root=args.nusc_root,
        nusc_version=args.nusc_version,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
    )
    caption_records, raw_records = load_caption_records(
        annotation_path=args.annotation_path,
        loader=scene_loader,
        prompt_template=args.prompt_template,
    )

    if not caption_records:
        raise ValueError("No caption records loaded from annotation file.")

    base_output_dir = ensure_dir(os.path.join(args.output_root, args.experiment_name))
    report_path = args.report_path or os.path.join(base_output_dir, "paper_report.json")

    for run in run_specs:
        stage_dir = ensure_dir(os.path.join(base_output_dir, run["stage_name"], "caption"))
        pred_path = os.path.join(stage_dir, "pred_caption.json")
        metrics_path = os.path.join(stage_dir, "caption_metrics.json")
        hallucination_path = os.path.join(stage_dir, "hallucination_eval.json")

        model, _ = init_model_for_eval(
            cfg_path=args.cfg_path,
            device=args.device,
            checkpoint=run["checkpoint"],
            second_checkpoint=run["second_checkpoint"],
            cfg_options=args.options,
        )

        dataset = CaptionDataset(caption_records, raw_records, scene_loader)
        dataloader = build_dataloader(dataset, args.batch_size, args.num_workers, caption_collate)
        predictions = generate_caption_predictions(
            model=model,
            dataloader=dataloader,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            min_length=args.min_length,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
        )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        save_json(
            pred_path,
            [
                {
                    "scene_id": item["scene_id"],
                    "prediction": item["prediction"],
                    "ground_truth": item["ground_truth"],
                }
                for item in predictions
            ],
        )

        metrics = evaluate_caption_predictions(
            predictions=predictions,
            bert_model=args.bert_model,
            bert_device=metric_device,
            bert_batch_size=args.bertscore_batch_size,
        )
        metrics.update(
            {
                "task_track": "caption",
                "evaluation_mode": "caption",
                "prompt_template": args.prompt_template,
                "decode": {
                    "temperature": 0.0,
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
        )
        hallucination_summary = build_hallucination_audit(
            predictions=predictions,
            sample_size=args.hallucination_sample_size,
            seed=args.seed,
            output_path=hallucination_path,
        )
        metrics["hallucination_audit"] = hallucination_summary
        save_json(metrics_path, metrics)

        update_paper_report(
            report_path=report_path,
            track="caption_track",
            stage_name=run["stage_name"],
            experiment_name=args.experiment_name,
            output_dir=stage_dir,
            metrics=metrics,
            checkpoint=run["checkpoint"] or "",
            second_checkpoint=run["second_checkpoint"] or "",
            ablation_tags=ablation_tags,
            hallucination_summary=hallucination_summary,
        )


if __name__ == "__main__":
    main()
