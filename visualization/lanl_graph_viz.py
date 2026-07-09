from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

import yaml


CONFIG_PATH = "configs/lanl_graph_inspection.yaml"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping.")

    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = [
        "input",
        "output",
        "inspection",
        "graph",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    required_input_keys = [
        "directory",
        "graph_nodes_filename",
        "graph_edges_filename",
    ]

    for key in required_input_keys:
        if key not in config["input"]:
            raise ValueError(f"Missing input.{key} in config.")

    required_output_keys = [
        "directory",
        "markdown_report_filename",
        "json_report_filename",
        "top_nodes_filename",
    ]

    for key in required_output_keys:
        if key not in config["output"]:
            raise ValueError(f"Missing output.{key} in config.")

    required_inspection_keys = [
        "top_k_nodes",
        "component_sample_size",
        "redteam_sample_size",
        "max_edges_for_visual_sample",
    ]

    for key in required_inspection_keys:
        if key not in config["inspection"]:
            raise ValueError(f"Missing inspection.{key} in config.")

    if "type" not in config["graph"]:
        raise ValueError("Missing graph.type in config.")

    if config["graph"]["type"] != "multidigraph":
        raise ValueError("Only graph.type=multidigraph is supported.")


def build_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_directory = Path(config["input"]["directory"])
    output_directory = Path(config["output"]["directory"])

    return {
        "graph_nodes": input_directory / config["input"]["graph_nodes_filename"],
        "graph_edges": input_directory / config["input"]["graph_edges_filename"],
        "markdown_report": output_directory
        / config["output"]["markdown_report_filename"],
        "json_report": output_directory
        / config["output"]["json_report_filename"],
        "top_nodes": output_directory / config["output"]["top_nodes_filename"],
    }


def validate_input_paths(paths: dict[str, Path]) -> None:
    required_inputs = [
        "graph_nodes",
        "graph_edges",
    ]

    for key in required_inputs:
        path = paths[key]

        if not path.exists():
            raise FileNotFoundError(f"Required input not found for {key}: {path}")

        if not path.is_file():
            raise FileNotFoundError(f"Required input is not a file for {key}: {path}")


def print_inspection_plan(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    print("LANL graph inspection plan")
    print("Graph type:", config["graph"]["type"])

    print("Inputs:")
    print(
        {
            "graph_nodes": str(paths["graph_nodes"]),
            "graph_edges": str(paths["graph_edges"]),
        }
    )

    print("Outputs:")
    print(
        {
            "markdown_report": str(paths["markdown_report"]),
            "json_report": str(paths["json_report"]),
            "top_nodes": str(paths["top_nodes"]),
        }
    )

    print("Inspection limits:")
    print(
        {
            "top_k_nodes": config["inspection"]["top_k_nodes"],
            "component_sample_size": config["inspection"]["component_sample_size"],
            "redteam_sample_size": config["inspection"]["redteam_sample_size"],
            "max_edges_for_visual_sample": config["inspection"][
                "max_edges_for_visual_sample"
            ],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to LANL graph inspection config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    validate_config(config)

    paths = build_paths(config)
    validate_input_paths(paths)

    output_directory = Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Config path:", args.config)
    print("Input directory:", config["input"]["directory"])
    print("Output directory:", output_directory)

    print_inspection_plan(
        config=config,
        paths=paths,
    )


if __name__ == "__main__":
    main()