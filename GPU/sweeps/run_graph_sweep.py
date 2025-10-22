"""Utility for launching the Bayesian hyperparameter sweep on Weights & Biases."""
from __future__ import annotations

import argparse
from pathlib import Path

import wandb
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="Weights & Biases entity under which the sweep should be created.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ddi-fp-graph",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("graph_bayesian.yaml"),
        help="Path to the sweep configuration YAML file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of agents to run sequentially after creating the sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        sweep_config = yaml.safe_load(handle)

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, count=args.count)


if __name__ == "__main__":
    main()
