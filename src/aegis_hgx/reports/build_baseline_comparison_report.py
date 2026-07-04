from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import yaml
from datetime import datetime, timezone


CONFIG_PATH = "configs/baseline_comparison_report.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Report config not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Report config must contain a YAML mapping.")

    return config


def load_json_artifact(artifact_path: str | Path) -> dict[str, Any]:
    path = Path(artifact_path)

    if not path.exists():
        raise FileNotFoundError(f"Required report artifact not found: {path}")

    with path.open(encoding="utf-8") as artifact_file:
        artifact = json.load(artifact_file)

    if not isinstance(artifact, dict):
        raise ValueError(f"Report artifact must be a JSON object: {path}")

    return artifact


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_path = config["inputs"]["calibration_metrics"]["path"]
    seed_stability_path = config["inputs"]["seed_stability"]["path"]

    calibration_artifact = load_json_artifact(calibration_path)
    seed_stability_artifact = load_json_artifact(seed_stability_path)

    return calibration_artifact, seed_stability_artifact


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


def extract_model_comparison_row(
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


def extract_model_comparison_rows(
    model_metric_artifacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []

    for model_key in sorted(model_metric_artifacts.keys()):
        row = extract_model_comparison_row(
            model_metric_artifacts[model_key]
        )
        rows.append(row)

    return rows


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def render_artifact_inventory(config: dict[str, Any]) -> str:
    lines = [
        "## Artifact Inventory",
        "",
        "| Artifact | Path | Purpose |",
        "|---|---|---|",
    ]

    for model_key, model_config in config["inputs"]["model_metrics"].items():
        lines.append(
            "| "
            f"{model_config['display_name']} "
            "| "
            f"`{model_config['path']}` "
            "| Baseline model metrics |"
        )

    calibration_config = config["inputs"]["calibration_metrics"]
    seed_config = config["inputs"]["seed_stability"]

    lines.append(
        "| "
        f"{calibration_config['display_name']} "
        "| "
        f"`{calibration_config['path']}` "
        "| Calibration evidence |"
    )
    lines.append(
        "| "
        f"{seed_config['display_name']} "
        "| "
        f"`{seed_config['path']}` "
        "| Seed-stability evidence |"
    )

    return "\n".join(lines)


def render_model_comparison_table(
    comparison_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "## Model Comparison",
        "",
        "| Model | Family | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in comparison_rows:
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

    return "\n".join(lines)


def render_initial_interpretation() -> str:
    return "\n".join(
        [
            "## Initial Interpretation",
            "",
            "This report compares the current CIC tabular baselines before the project moves into graph construction and GNN modeling.",
            "",
            "PR-AUC is treated as the most important ranking metric because cyber attack detection is usually class-imbalanced. ROC-AUC remains useful for separability, but it can look optimistic when the negative class dominates.",
            "",
            "Precision, recall, F1, false positives, and false negatives are threshold-dependent metrics. They describe what happens when model scores are converted into operational alerts.",
            "",
            "A future graph model should not be considered better just because it is more complex. It should beat the strongest tabular baseline on meaningful metrics such as PR-AUC, false-positive behavior, false-negative behavior, calibration quality, and stability.",
        ]
    )


def render_report(
    config: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    calibration_artifact: dict[str, Any],
    seed_stability_artifact: dict[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()

    sections = [
        f"# {config['report']['title']}",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Purpose",
        "",
        "This report consolidates the CIC baseline laboratory results and defines the performance bar that future graph models must beat.",
        "",
        "## Phase Classification",
        "",
        "| Topic | Phase | Reason |",
        "|---|---|---|",
        "| Baseline comparison report | Training/evaluation/test | Summarizes offline experiment evidence |",
        "| Model metric artifacts | Training/evaluation/test | Produced by prior offline training runs |",
        "| Calibration evidence | Training/evaluation/test | Evaluates score trustworthiness on held-out data |",
        "| Seed-stability evidence | Training/evaluation/test | Measures variability across repeated offline runs |",
        "| Inference API | Not changed | No serving code is modified by this report |",
        "",
        render_artifact_inventory(config),
        "",
        render_model_comparison_table(comparison_rows),
        "",
        render_initial_interpretation(),
        "",
        render_calibration_summary(calibration_artifact),
        "",
        render_seed_stability_summary(seed_stability_artifact),
        "",
        render_graph_target_bar(comparison_rows),
        "",
        render_limitations(),
        "",
        render_next_steps(),
        "",
    ]

    return "\n".join(sections)


def write_report(report_markdown: str, config: dict[str, Any]) -> Path:
    output_path = Path(config["report"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown, encoding="utf-8")
    return output_path


def format_interval(
    lower_value: Any,
    upper_value: Any,
) -> str:
    lower = format_metric(lower_value)
    upper = format_metric(upper_value)

    if lower == "N/A" or upper == "N/A":
        return "N/A"

    return f"[{lower}, {upper}]"


def render_calibration_summary(
    calibration_artifact: dict[str, Any],
) -> str:
    calibration = calibration_artifact.get("calibration", {})
    probability_summary = calibration_artifact.get(
        "probability_summary",
        {},
    )

    brier_score = calibration.get("brier_score")
    expected_calibration_error = calibration.get(
        "expected_calibration_error"
    )
    n_bins = calibration.get("n_bins")
    strategy = calibration.get("strategy")

    lines = [
        "## Calibration Summary",
        "",
        "Calibration evaluates whether model scores can be interpreted as trustworthy risk probabilities.",
        "",
        "| Metric | Value | Interpretation |",
        "|---|---:|---|",
        (
            "| Brier score | "
            f"{format_metric(brier_score)} "
            "| Lower is better; measures squared probability error |"
        ),
        (
            "| Expected calibration error | "
            f"{format_metric(expected_calibration_error)} "
            "| Lower is better; measures weighted bin-level calibration gap |"
        ),
        (
            "| Calibration bins | "
            f"{n_bins if n_bins is not None else 'N/A'} "
            "| Number of probability buckets used in the reliability analysis |"
        ),
        (
            "| Strategy | "
            f"{strategy if strategy is not None else 'N/A'} "
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

    return "\n".join(lines)


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


def find_best_row(
    comparison_rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any] | None:
    eligible_rows = [
        row
        for row in comparison_rows
        if row.get(metric_name) is not None
    ]

    if not eligible_rows:
        return None

    return max(
        eligible_rows,
        key=lambda row: float(row[metric_name]),
    )


def render_graph_target_bar(
    comparison_rows: list[dict[str, Any]],
) -> str:
    best_pr_auc_row = find_best_row(comparison_rows, "pr_auc")
    best_f1_row = find_best_row(comparison_rows, "f1")

    best_pr_auc_text = (
        (
            f"{best_pr_auc_row['display_name']} "
            f"with PR-AUC {format_metric(best_pr_auc_row['pr_auc'])}"
        )
        if best_pr_auc_row is not None
        else "N/A"
    )
    best_f1_text = (
        (
            f"{best_f1_row['display_name']} "
            f"with F1 {format_metric(best_f1_row['f1'])}"
        )
        if best_f1_row is not None
        else "N/A"
    )

    return "\n".join(
        [
            "## Graph-Model Target Bar",
            "",
            "Future graph models must justify their additional complexity. A GNN should not be treated as better simply because it is more advanced.",
            "",
            "| Target | Current Baseline Bar |",
            "|---|---|",
            f"| Strongest PR-AUC baseline | {best_pr_auc_text} |",
            f"| Strongest F1 baseline | {best_f1_text} |",
            "",
            "A future graph model should aim to improve at least one of the following without severely degrading the others:",
            "",
            "- Higher PR-AUC on held-out data.",
            "- Better false-positive behavior at useful recall levels.",
            "- Better false-negative behavior for attack traffic.",
            "- More stable results across random seeds.",
            "- Better calibration or more honest risk scores.",
            "- More explainable relationship-level anomaly evidence.",
            "",
            "The strongest future claim will not be: `the graph model is more complex`. The stronger claim will be: `the graph model captures relationship structure that tabular baselines miss, and the evidence shows measurable improvement.`",
        ]
    )


def render_limitations() -> str:
    return "\n".join(
        [
            "## Limitations",
            "",
            "The current baseline laboratory is intentionally limited.",
            "",
            "- The current comparison is based on CIC tabular flow features.",
            "- The current split strategy is random rather than temporal.",
            "- The current graph structure has not yet been constructed.",
            "- The current calibration analysis focuses on the PyTorch MLP baseline.",
            "- The current multi-seed stability analysis focuses on the PyTorch MLP baseline.",
            "- The current report does not yet include graph baselines, temporal models, adversarial robustness, or drift analysis.",
            "",
            "These limitations are acceptable at this stage because the purpose of this milestone is to establish a serious tabular baseline bar before graph modeling begins.",
        ]
    )


def render_next_steps() -> str:
    return "\n".join(
        [
            "## Next Steps",
            "",
            "The next milestone is Release 2: Baseline Laboratory.",
            "",
            "Immediate next steps:",
            "",
            "- Finalize the baseline laboratory release report.",
            "- Confirm that all baseline artifacts are reproducible through CI.",
            "- Use this report to define the minimum bar for graph models.",
            "- Begin graph construction notes and graph schema design.",
            "- Move from isolated tabular rows toward entities, edges, and relationship structure.",
            "",
            "The next modeling phase should only claim graph-model value if graph structure produces measurable improvement over this baseline evidence.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to baseline comparison report config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_metric_artifacts = load_model_metric_artifacts(config)
    calibration_artifact, seed_stability_artifact = load_auxiliary_artifacts(
        config
    )
    comparison_rows = extract_model_comparison_rows(
        model_metric_artifacts
    )
    report_markdown = render_report(
        config=config,
        comparison_rows=comparison_rows,
        calibration_artifact=calibration_artifact,
        seed_stability_artifact=seed_stability_artifact,
    )
    report_path = write_report(
        report_markdown=report_markdown,
        config=config,
    )

    print("Config path:", args.config)
    print("Report title:", config["report"]["title"])
    print("Output path:", config["report"]["output_path"])

    print("Model metric artifacts:")
    for model_key, model_config in config["inputs"]["model_metrics"].items():
        print(
            "-",
            model_key,
            model_config["path"],
            model_config["display_name"],
        )

    print(
        "Calibration metrics path:",
        config["inputs"]["calibration_metrics"]["path"],
    )
    print(
        "Seed stability path:",
        config["inputs"]["seed_stability"]["path"],
    )
    print("Loaded model artifact count:", len(model_metric_artifacts))
    print("Loaded model artifact keys:", sorted(model_metric_artifacts.keys()))
    print(
        "Calibration artifact keys:",
        sorted(calibration_artifact.keys()),
    )
    print(
        "Seed stability artifact keys:",
        sorted(seed_stability_artifact.keys()),
    )
    print("Extracted comparison rows:", len(comparison_rows))

    for row in comparison_rows:
        print(
            "Comparison row:",
            {
                "model_key": row["model_key"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "f1": row["f1"],
            },
        )
    print("Final report path:", report_path)


if __name__ == "__main__":
    main()