from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json

import yaml


CONFIG_PATH = "configs/release_2_baseline_laboratory.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Release config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Release config must contain a YAML mapping.")

    return config


def load_json_artifact(artifact_path: str | Path) -> dict[str, Any]:
    path = Path(artifact_path)

    if not path.exists():
        raise FileNotFoundError(f"Required release artifact not found: {path}")

    with path.open(encoding="utf-8") as artifact_file:
        artifact = json.load(artifact_file)

    if not isinstance(artifact, dict):
        raise ValueError(f"Release artifact must be a JSON object: {path}")

    return artifact


def load_text_artifact(artifact_path: str | Path) -> str:
    path = Path(artifact_path)

    if not path.exists():
        raise FileNotFoundError(f"Required release artifact not found: {path}")

    return path.read_text(encoding="utf-8")


def load_model_metric_artifacts(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    loaded_artifacts = {}

    for model_key, model_config in config["inputs"]["model_metrics"].items():
        artifact = load_json_artifact(model_config["path"])

        loaded_artifacts[model_key] = {
            "model_key": model_key,
            "display_name": model_config["display_name"],
            "model_family": model_config["model_family"],
            "path": model_config["path"],
            "artifact": artifact,
        }

    return loaded_artifacts


def load_auxiliary_artifacts(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    calibration_artifact = load_json_artifact(
        config["inputs"]["calibration_metrics"]["path"]
    )
    seed_stability_artifact = load_json_artifact(
        config["inputs"]["seed_stability"]["path"]
    )
    baseline_comparison_report = load_text_artifact(
        config["inputs"]["baseline_comparison_report"]["path"]
    )

    return (
        calibration_artifact,
        seed_stability_artifact,
        baseline_comparison_report,
    )


def get_nested_value(
    data: dict[str, Any],
    path: tuple[str, ...],
) -> Any | None:
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return None

        if key not in current:
            return None

        current = current[key]

    return current


def extract_first_available(
    artifact: dict[str, Any],
    candidate_paths: list[tuple[str, ...]],
) -> Any | None:
    for path in candidate_paths:
        value = get_nested_value(artifact, path)

        if value is not None:
            return value

    return None


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def format_interval(
    lower_value: Any,
    upper_value: Any,
) -> str:
    lower = format_metric(lower_value)
    upper = format_metric(upper_value)

    if lower == "N/A" or upper == "N/A":
        return "N/A"

    return f"[{lower}, {upper}]"


def extract_model_summary_row(
    model_artifact: dict[str, Any],
) -> dict[str, Any]:
    artifact = model_artifact["artifact"]

    accuracy = extract_first_available(
        artifact,
        [
            ("accuracy",),
            ("default_threshold_metrics", "test_accuracy"),
        ],
    )
    precision = extract_first_available(
        artifact,
        [
            ("precision",),
            ("default_threshold_metrics", "test_precision"),
        ],
    )
    recall = extract_first_available(
        artifact,
        [
            ("recall",),
            ("default_threshold_metrics", "test_recall"),
        ],
    )
    f1 = extract_first_available(
        artifact,
        [
            ("f1",),
            ("default_threshold_metrics", "test_f1"),
        ],
    )
    roc_auc = extract_first_available(
        artifact,
        [
            ("roc_auc",),
            ("threshold_independent_metrics", "test_roc_auc"),
        ],
    )
    pr_auc = extract_first_available(
        artifact,
        [
            ("pr_auc",),
            ("threshold_independent_metrics", "test_pr_auc"),
        ],
    )
    true_positive = extract_first_available(
        artifact,
        [
            ("true_positive",),
            ("default_threshold_metrics", "test_true_positive"),
        ],
    )
    false_positive = extract_first_available(
        artifact,
        [
            ("false_positive",),
            ("default_threshold_metrics", "test_false_positive"),
        ],
    )
    true_negative = extract_first_available(
        artifact,
        [
            ("true_negative",),
            ("default_threshold_metrics", "test_true_negative"),
        ],
    )
    false_negative = extract_first_available(
        artifact,
        [
            ("false_negative",),
            ("default_threshold_metrics", "test_false_negative"),
        ],
    )

    return {
        "model_key": model_artifact["model_key"],
        "display_name": model_artifact["display_name"],
        "model_family": model_artifact["model_family"],
        "accuracy": to_float_or_none(accuracy),
        "precision": to_float_or_none(precision),
        "recall": to_float_or_none(recall),
        "f1": to_float_or_none(f1),
        "roc_auc": to_float_or_none(roc_auc),
        "pr_auc": to_float_or_none(pr_auc),
        "true_positive": to_float_or_none(true_positive),
        "false_positive": to_float_or_none(false_positive),
        "true_negative": to_float_or_none(true_negative),
        "false_negative": to_float_or_none(false_negative),
        "artifact_path": model_artifact["path"],
    }


def extract_model_summary_rows(
    model_metric_artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for model_key in sorted(model_metric_artifacts.keys()):
        rows.append(
            extract_model_summary_row(model_metric_artifacts[model_key])
        )

    return rows


def find_best_row(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any] | None:
    eligible_rows = [
        row
        for row in rows
        if row.get(metric_name) is not None
    ]

    if not eligible_rows:
        return None

    return max(eligible_rows, key=lambda row: float(row[metric_name]))


def render_release_summary(config: dict[str, Any]) -> str:
    return "\n".join(
        [
            "## Release Summary",
            "",
            "This release closes the CIC baseline laboratory milestone for AEGIS-HGX.",
            "",
            "The baseline laboratory moved the project from synthetic experiments into public cyber intrusion-detection data and established a serious tabular-model evidence base before graph modeling begins.",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Phase name | {config['release']['phase_name']} |",
            f"| Dataset | {config['release']['dataset']} |",
            "| Primary phase | Training/evaluation/test reporting |",
            "| Inference impact | Indirect; no serving code changed |",
            "",
            "The release is complete when ingestion, tabular feature construction, baseline training, training diagnostics, calibration analysis, seed-stability analysis, and baseline comparison are all represented by reproducible artifacts.",
        ]
    )


def render_phase_classification() -> str:
    return "\n".join(
        [
            "## Phase Classification",
            "",
            "| Component | Phase | Reason |",
            "|---|---|---|",
            "| CIC ingestion | Data preparation | Converts public CIC files into project-ready data |",
            "| Tabular feature construction | Training/evaluation preparation | Produces model-ready features |",
            "| Logistic, XGBoost, and MLP baselines | Training/evaluation/test | Trains and evaluates offline models |",
            "| PyTorch MLP diagnostics | Training/evaluation/test | Captures training behavior and threshold behavior |",
            "| Calibration analysis | Training/evaluation/test | Evaluates score trustworthiness on held-out data |",
            "| Seed-stability analysis | Training/evaluation/test | Measures run-to-run variability |",
            "| Baseline comparison report | Training/evaluation/test reporting | Summarizes offline evidence |",
            "| Inference API | Not changed | This release does not modify serving behavior |",
        ]
    )


def render_artifact_inventory(config: dict[str, Any]) -> str:
    lines = [
        "## Artifact Inventory",
        "",
        "| Artifact | Path | Purpose |",
        "|---|---|---|",
    ]

    for model_config in config["inputs"]["model_metrics"].values():
        lines.append(
            "| "
            f"{model_config['display_name']} "
            "| "
            f"`{model_config['path']}` "
            "| Baseline model metrics |"
        )

    calibration_config = config["inputs"]["calibration_metrics"]
    seed_config = config["inputs"]["seed_stability"]
    comparison_config = config["inputs"]["baseline_comparison_report"]

    lines.append(
        "| "
        f"{calibration_config['display_name']} "
        "| "
        f"`{calibration_config['path']}` "
        "| Score calibration evidence |"
    )
    lines.append(
        "| "
        f"{seed_config['display_name']} "
        "| "
        f"`{seed_config['path']}` "
        "| Multi-seed stability evidence |"
    )
    lines.append(
        "| "
        f"{comparison_config['display_name']} "
        "| "
        f"`{comparison_config['path']}` "
        "| Baseline comparison and graph-target bar |"
    )

    return "\n".join(lines)


def render_implementation_journey() -> str:
    return "\n".join(
        [
            "## Implementation Journey",
            "",
            "This release covers the complete baseline laboratory path:",
            "",
            "1. Public CIC data was ingested into the project.",
            "2. Raw CIC flow columns were normalized into clean tabular features.",
            "3. A logistic regression baseline established a simple linear reference point.",
            "4. An XGBoost baseline established a strong tabular tree-based reference point.",
            "5. A scikit-learn MLP baseline established a quick neural-network reference point.",
            "6. A PyTorch MLP diagnostic trainer exposed the training loop, validation behavior, threshold behavior, PR/ROC curves, and model lineage.",
            "7. Calibration analysis tested whether raw model scores behaved like trustworthy risk probabilities.",
            "8. Multi-seed stability analysis tested whether PyTorch MLP results were consistent across random seeds.",
            "9. A baseline comparison report synthesized the evidence and defined what future graph models must beat.",
        ]
    )


def render_model_summary(model_rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Baseline Model Summary",
        "",
        "| Model | Family | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in model_rows:
        lines.append(
            "| "
            f"{row['display_name']} "
            "| "
            f"{row['model_family']} "
            "| "
            f"{format_metric(row['pr_auc'])} "
            "| "
            f"{format_metric(row['roc_auc'])} "
            "| "
            f"{format_metric(row['f1'])} "
            "| "
            f"{format_metric(row['precision'])} "
            "| "
            f"{format_metric(row['recall'])} "
            "| "
            f"{format_metric(row['accuracy'])} "
            "| "
            f"{format_metric(row['false_positive'])} "
            "| "
            f"{format_metric(row['false_negative'])} "
            "|"
        )

    best_pr_auc = find_best_row(model_rows, "pr_auc")
    best_f1 = find_best_row(model_rows, "f1")

    best_pr_auc_text = (
        (
            f"{best_pr_auc['display_name']} "
            f"with PR-AUC {format_metric(best_pr_auc['pr_auc'])}"
        )
        if best_pr_auc is not None
        else "N/A"
    )
    best_f1_text = (
        (
            f"{best_f1['display_name']} "
            f"with F1 {format_metric(best_f1['f1'])}"
        )
        if best_f1 is not None
        else "N/A"
    )

    lines.extend(
        [
            "",
            "### Baseline Bar",
            "",
            f"- Strongest PR-AUC baseline: {best_pr_auc_text}.",
            f"- Strongest F1 baseline: {best_f1_text}.",
            "",
            "PR-AUC is treated as especially important because cyber attack detection is usually class-imbalanced. F1, precision, recall, false positives, and false negatives describe thresholded alert behavior.",
        ]
    )

    return "\n".join(lines)


def render_calibration_summary(
    calibration_artifact: dict[str, Any],
) -> str:
    calibration = calibration_artifact.get("calibration", {})
    probability_summary = calibration_artifact.get(
        "probability_summary",
        {},
    )

    return "\n".join(
        [
            "## Calibration Summary",
            "",
            "Calibration analysis evaluates whether model scores can be interpreted as trustworthy risk probabilities.",
            "",
            "| Metric | Value | Interpretation |",
            "|---|---:|---|",
            (
                "| Brier score | "
                f"{format_metric(calibration.get('brier_score'))} "
                "| Lower is better; measures squared probability error |"
            ),
            (
                "| Expected calibration error | "
                f"{format_metric(calibration.get('expected_calibration_error'))} "
                "| Lower is better; measures weighted bin-level calibration gap |"
            ),
            (
                "| Calibration bins | "
                f"{calibration.get('n_bins', 'N/A')} "
                "| Number of probability buckets used in reliability analysis |"
            ),
            (
                "| Strategy | "
                f"{calibration.get('strategy', 'N/A')} "
                "| Probability-bin construction strategy |"
            ),
            "",
            "### Probability Summary",
            "",
            "| Statistic | Value |",
            "|---|---:|",
            (
                "| Minimum probability | "
                f"{format_metric(probability_summary.get('min_probability'))} |"
            ),
            (
                "| Maximum probability | "
                f"{format_metric(probability_summary.get('max_probability'))} |"
            ),
            (
                "| Mean probability | "
                f"{format_metric(probability_summary.get('mean_probability'))} |"
            ),
            (
                "| Median probability | "
                f"{format_metric(probability_summary.get('median_probability'))} |"
            ),
            "",
            "Calibration is separate from ranking. A model can have strong PR-AUC and ROC-AUC while still producing poorly calibrated probability scores.",
        ]
    )


def render_seed_stability_summary(
    seed_stability_artifact: dict[str, Any],
) -> str:
    metrics = seed_stability_artifact.get("metrics", {})

    metric_names = [
        "pr_auc",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "false_positive",
        "false_negative",
    ]

    lines = [
        "## Seed-Stability Summary",
        "",
        "Seed-stability analysis evaluates whether the PyTorch MLP baseline is reliable across repeated training runs.",
        "",
        "| Metric | Mean | Std | 95% CI | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for metric_name in metric_names:
        metric_summary = metrics.get(metric_name, {})

        lines.append(
            "| "
            f"{metric_name} "
            "| "
            f"{format_metric(metric_summary.get('mean'))} "
            "| "
            f"{format_metric(metric_summary.get('std'))} "
            "| "
            f"{format_interval(metric_summary.get('ci95_lower'), metric_summary.get('ci95_upper'))} "
            "| "
            f"{format_metric(metric_summary.get('min'))} "
            "| "
            f"{format_metric(metric_summary.get('max'))} "
            "|"
        )

    lines.extend(
        [
            "",
            "Low standard deviation means the baseline is stable across seeds. High standard deviation means the model may be sensitive to train/test split, initialization, dropout, batch ordering, or optimizer trajectory.",
        ]
    )

    return "\n".join(lines)


def render_baseline_comparison_summary(
    baseline_comparison_report: str,
) -> str:
    available = bool(baseline_comparison_report.strip())

    return "\n".join(
        [
            "## Baseline Comparison Summary",
            "",
            (
                "The baseline comparison report was generated and is available as a release input."
                if available
                else "The baseline comparison report was configured but appears to be empty."
            ),
            "",
            "The key conclusion is that future graph models must justify their additional complexity. A GNN should not be treated as better simply because it is more advanced; it must show measurable improvement over tabular baselines.",
        ]
    )


def render_graph_readiness(model_rows: list[dict[str, Any]]) -> str:
    best_pr_auc = find_best_row(model_rows, "pr_auc")
    best_f1 = find_best_row(model_rows, "f1")

    best_pr_auc_text = (
        (
            f"{best_pr_auc['display_name']} "
            f"with PR-AUC {format_metric(best_pr_auc['pr_auc'])}"
        )
        if best_pr_auc is not None
        else "N/A"
    )
    best_f1_text = (
        (
            f"{best_f1['display_name']} "
            f"with F1 {format_metric(best_f1['f1'])}"
        )
        if best_f1 is not None
        else "N/A"
    )

    return "\n".join(
        [
            "## Graph-Readiness Statement",
            "",
            "The project is ready to begin graph construction because the tabular baseline laboratory has established a measurable baseline bar.",
            "",
            "| Target | Current Baseline Bar |",
            "|---|---|",
            f"| Strongest PR-AUC baseline | {best_pr_auc_text} |",
            f"| Strongest F1 baseline | {best_f1_text} |",
            "",
            "Future graph models should aim to improve at least one of the following without severely degrading the others:",
            "",
            "- PR-AUC on held-out data.",
            "- False-positive behavior at useful recall levels.",
            "- False-negative behavior for attack traffic.",
            "- Seed stability.",
            "- Calibration quality.",
            "- Relationship-level anomaly evidence.",
            "",
            "The next phase should move from isolated tabular flow rows toward nodes, edges, entity relationships, and graph-structured evidence.",
        ]
    )


def render_limitations() -> str:
    return "\n".join(
        [
            "## Limitations",
            "",
            "The baseline laboratory is complete for its intended purpose, but it is intentionally limited.",
            "",
            "- The release focuses on CIC tabular flow features.",
            "- The release does not yet include graph construction.",
            "- The release does not yet include temporal train/test splitting.",
            "- The release does not yet include graph neural networks.",
            "- Calibration and seed-stability evidence are currently strongest for the PyTorch MLP baseline.",
            "- The release does not yet include adversarial robustness, drift evaluation, or production-scale profiling.",
            "",
            "These limitations are acceptable because this milestone is meant to close the tabular baseline lab before the graph research phase begins.",
        ]
    )


def render_next_phase_handoff() -> str:
    return "\n".join(
        [
            "## Next-Phase Handoff",
            "",
            "The next phase begins graph construction from events.",
            "",
            "The immediate next objectives are:",
            "",
            "- Define graph construction concepts from first principles.",
            "- Decide which CIC entities become nodes.",
            "- Decide which flow relationships become edges.",
            "- Build node and edge tables.",
            "- Add graph inspection and visualization.",
            "- Convert the graph into model-ready structures for future GNN baselines.",
            "",
        ]
    )


def render_release_report(
    config: dict[str, Any],
    model_rows: list[dict[str, Any]],
    calibration_artifact: dict[str, Any],
    seed_stability_artifact: dict[str, Any],
    baseline_comparison_report: str,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()

    sections = [
        f"# {config['release']['title']}",
        "",
        f"Generated at: `{generated_at}`",
        "",
        render_release_summary(config),
        "",
        render_phase_classification(),
        "",
        render_artifact_inventory(config),
        "",
        render_implementation_journey(),
        "",
        render_model_summary(model_rows),
        "",
        render_calibration_summary(calibration_artifact),
        "",
        render_seed_stability_summary(seed_stability_artifact),
        "",
        render_baseline_comparison_summary(baseline_comparison_report),
        "",
        render_graph_readiness(model_rows),
        "",
        render_limitations(),
        "",
        render_next_phase_handoff(),
        "",
    ]

    return "\n".join(sections)


def write_release_report(
    report_markdown: str,
    config: dict[str, Any],
) -> Path:
    output_path = Path(config["release"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to baseline laboratory release config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_metric_artifacts = load_model_metric_artifacts(config)
    (
        calibration_artifact,
        seed_stability_artifact,
        baseline_comparison_report,
    ) = load_auxiliary_artifacts(config)

    model_rows = extract_model_summary_rows(model_metric_artifacts)

    release_markdown = render_release_report(
        config=config,
        model_rows=model_rows,
        calibration_artifact=calibration_artifact,
        seed_stability_artifact=seed_stability_artifact,
        baseline_comparison_report=baseline_comparison_report,
    )
    release_path = write_release_report(
        report_markdown=release_markdown,
        config=config,
    )

    print("Config path:", args.config)
    print("Release title:", config["release"]["title"])
    print("Release path:", release_path)
    print("Loaded model artifact count:", len(model_metric_artifacts))
    print("Model rows:", len(model_rows))
    print("Calibration artifact:", config["inputs"]["calibration_metrics"]["path"])
    print("Seed stability artifact:", config["inputs"]["seed_stability"]["path"])
    print(
        "Baseline comparison report:",
        config["inputs"]["baseline_comparison_report"]["path"],
    )


if __name__ == "__main__":
    main()