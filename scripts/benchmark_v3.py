#!/usr/bin/env python3
"""Benchmark HerdingEnv-v3 models and heuristics on a fixed evaluation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import shepherding.envs  # noqa: F401

from shepherding.research import (
    aggregate_results,
    create_significance_table,
    load_yaml_config,
    run_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suite for HerdingEnv-v3")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/research/benchmark_v3.yaml",
    )
    parser.add_argument("--output-dir", type=str, default="results/research_v3/benchmarks")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    benchmark_cfg = load_yaml_config(Path(args.benchmark_config))["benchmark"]
    output_dir = Path(args.output_dir)

    run_benchmark(
        config=config,
        benchmark_cfg=benchmark_cfg,
        output_dir=output_dir,
        save_models=args.save_models,
    )
    aggregate_results(
        output_dir / "episode_summaries.csv",
        output_dir / "aggregate_metrics.csv",
    )
    create_significance_table(
        output_dir / "episode_summaries.csv",
        output_dir / "significance_tests.csv",
    )
    print(f"Saved benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
