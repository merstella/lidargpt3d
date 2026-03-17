import json
from pathlib import Path
from typing import Any, Dict


def _load_report(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "caption_track": {},
        "qa_track": {},
    }


def _hallucination_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    flagged = payload.get("flagged_count")
    pending = payload.get("pending_count")
    return {
        "sampled_scenes": payload.get("sampled_scenes", 0),
        "flagged_count": flagged,
        "pending_count": pending,
        "source_file": payload.get("path"),
    }


def update_paper_report(
    report_path: str,
    track: str,
    stage_name: str,
    experiment_name: str,
    output_dir: str,
    metrics: Dict[str, Any],
    checkpoint: str,
    second_checkpoint: str = "",
    ablation_tags: Dict[str, str] = None,
    hallucination_summary: Dict[str, Any] = None,
):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = _load_report(path)
    report["experiment_name"] = experiment_name

    payload: Dict[str, Any] = {
        "stage_name": stage_name,
        "checkpoint": checkpoint,
        "second_checkpoint": second_checkpoint,
        "output_dir": output_dir,
        "ablation_tags": ablation_tags or {},
        "metrics": metrics,
    }
    if hallucination_summary is not None:
        payload["hallucination_audit_summary"] = _hallucination_summary(hallucination_summary)

    report.setdefault(track, {})
    report[track][stage_name] = payload

    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    md_path = path.with_suffix(".md")
    md_path.write_text(render_report_markdown(report), encoding="utf-8")


def render_report_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Evaluation Report: {report.get('experiment_name', 'unnamed')}"]

    lines.append("")
    lines.append("## Caption Track")
    if report.get("caption_track"):
        for stage_name, payload in sorted(report["caption_track"].items()):
            metrics = payload["metrics"]
            lines.append(f"### {stage_name}")
            lines.append(f"- BLEU-1: {metrics.get('bleu-1', 0.0):.4f}")
            lines.append(f"- BERTScore-F1: {metrics.get('bertscore_f1', 0.0):.4f}")
            lines.append(f"- ROUGE-L: {metrics.get('rouge-l', 0.0):.4f}")
            lines.append(f"- CIDEr: {metrics.get('cider', 0.0):.4f}")
            hallucination = payload.get("hallucination_audit_summary", {})
            if hallucination:
                lines.append(
                    "- Hallucination audit: "
                    f"sampled={hallucination.get('sampled_scenes', 0)}, "
                    f"flagged={hallucination.get('flagged_count')}, "
                    f"pending={hallucination.get('pending_count')}"
                )
    else:
        lines.append("- No caption runs recorded.")

    lines.append("")
    lines.append("## QA Track")
    if report.get("qa_track"):
        for stage_name, payload in sorted(report["qa_track"].items()):
            metrics = payload["metrics"]
            lines.append(f"### {stage_name}")
            lines.append(f"- Exact Match: {metrics.get('exact_match', 0.0):.4f}")
            lines.append(f"- Overall Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            lines.append("- Accuracy by question type:")
            for key, value in metrics.get("accuracy_by_question_type", {}).items():
                lines.append(f"  - {key}: {value.get('accuracy', 0.0):.4f} ({value.get('num_samples', 0)})")
            lines.append("- Accuracy by answer type:")
            for key, value in metrics.get("accuracy_by_answer_type", {}).items():
                lines.append(f"  - {key}: {value.get('accuracy', 0.0):.4f} ({value.get('num_samples', 0)})")
    else:
        lines.append("- No QA runs recorded.")

    lines.append("")
    return "\n".join(lines)

