import argparse
import gc
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics.common import (
    QA_PROMPT_TEMPLATE,
    QADataset,
    SceneCacheLoader,
    build_dataloader,
    build_run_specs,
    ensure_dir,
    generate_qa_predictions,
    init_model_for_eval,
    load_qa_records,
    parse_ablation_tags,
    qa_collate,
    save_json,
)
from evaluation.metrics.qa_metrics import evaluate_qa_predictions
from evaluation.metrics.reporting import update_paper_report


def parse_args():
    parser = argparse.ArgumentParser(description="QA evaluation for NuScenes-QA.")
    parser.add_argument("--cfg-path", default="./eval_configs/benchmark_evaluation_nuscenes.yaml")
    parser.add_argument("--annotation-path", required=True, help="NuScenes-QA val/test json/jsonl")
    parser.add_argument("--scene-cache", required=True, help="Scene cache directory or raw nuScenes root")
    parser.add_argument("--nusc-root", default=None)
    parser.add_argument("--nusc-version", default="v1.0-trainval")
    parser.add_argument("--output-root", default="evaluation/outputs")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--stage-name", default="stage4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--second-checkpoint", default=None)
    parser.add_argument("--checkpoint-spec", nargs="*", default=None, help="Repeated stage=checkpoint")
    parser.add_argument("--second-checkpoint-spec", nargs="*", default=None, help="Repeated stage=checkpoint")
    parser.add_argument("--prompt-template", default=QA_PROMPT_TEMPLATE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--num-sweeps", type=int, default=1)
    parser.add_argument("--normalize-pc", action="store_true")
    parser.add_argument("--scan-scene-cache", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--ablation-tag", action="append", default=[])
    parser.add_argument("--options", nargs="+", default=None, help="Extra model cfg overrides")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature != 0.0 or args.do_sample or args.num_beams != 1:
        raise ValueError(
            "QA evaluation must use deterministic decoding with temperature=0 and num_beams=1."
        )
    run_specs = build_run_specs(
        stage_name=args.stage_name,
        checkpoint=args.checkpoint,
        second_checkpoint=args.second_checkpoint,
        checkpoint_spec=args.checkpoint_spec,
        second_checkpoint_spec=args.second_checkpoint_spec,
        allowed_stages=("stage3", "stage4"),
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
    )
    qa_records, raw_records = load_qa_records(
        annotation_path=args.annotation_path,
        loader=scene_loader,
        prompt_template=args.prompt_template,
    )

    if not qa_records:
        raise ValueError("No QA records loaded from annotation file.")

    base_output_dir = ensure_dir(os.path.join(args.output_root, args.experiment_name))
    report_path = args.report_path or os.path.join(base_output_dir, "paper_report.json")

    for run in run_specs:
        stage_dir = ensure_dir(os.path.join(base_output_dir, run["stage_name"], "qa"))
        pred_path = os.path.join(stage_dir, "pred_qa.json")
        metrics_path = os.path.join(stage_dir, "qa_metrics.json")

        model, _ = init_model_for_eval(
            cfg_path=args.cfg_path,
            device=args.device,
            checkpoint=run["checkpoint"],
            second_checkpoint=run["second_checkpoint"],
            cfg_options=args.options,
        )

        dataset = QADataset(qa_records, raw_records, scene_loader)
        dataloader = build_dataloader(dataset, args.batch_size, args.num_workers, qa_collate)
        predictions = generate_qa_predictions(
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
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "prediction": item["prediction"],
                    "ground_truth": item["ground_truth"],
                }
                for item in predictions
            ],
        )

        metrics = evaluate_qa_predictions(predictions)
        scored_predictions = metrics.pop("scored_predictions")
        metrics.update(
            {
                "task_track": "qa",
                "evaluation_mode": "qa_probing" if run["stage_name"].lower() == "stage3" else "qa_full",
                "prompt_template": args.prompt_template,
                "decode": {
                    "temperature": 0.0,
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
        )
        save_json(metrics_path, metrics)

        update_paper_report(
            report_path=report_path,
            track="qa_track",
            stage_name=run["stage_name"],
            experiment_name=args.experiment_name,
            output_dir=stage_dir,
            metrics=metrics,
            checkpoint=run["checkpoint"] or "",
            second_checkpoint=run["second_checkpoint"] or "",
            ablation_tags=ablation_tags,
        )

        detailed_path = os.path.join(stage_dir, "pred_qa_scored.json")
        save_json(detailed_path, scored_predictions)


if __name__ == "__main__":
    main()
