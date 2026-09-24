#!/usr/bin/env python3
"""Score a trained v3 agent on the held-out procedural test suite.

The five hand-authored ``unseen_*`` presets are only five points, and every
hyperparameter tuned against them leaks into the reported generalization number.
This entry point instead evaluates on a large sample of layouts drawn from a
distribution that is disjoint from the training ranges, and reports graded
delivery metrics with bootstrap confidence intervals alongside the strict
all-or-nothing success rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch

import shepherding.envs  # noqa: F401

from shepherding.baselines import HeuristicShepherdAgent
from shepherding.research import (
    evaluate_scenarios,
    load_model,
    load_obs_normalizer,
    load_yaml_config,
    save_rows,
    save_summaries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out generalization evaluation")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Required for every model type except the heuristic.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="recurrent",
        choices=["recurrent", "feedforward", "behavioral_cloning", "heuristic"],
    )
    parser.add_argument(
        "--fixed-sheep-count",
        action="store_true",
        help="Hold the flock at the configured n_sheep. The heuristic and cloning "
        "agents always run this way; pass it to a PPO model to compare on "
        "identical episodes.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to the VecNormalize statistics saved next to the model.",
    )
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        default=None,
        help="Defaults to the procedural test suite plus the named unseen presets.",
    )
    parser.add_argument("--output-dir", type=str, default="results/generalization")
    return parser.parse_args()


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def main() -> None:
    args = parse_args()
    # Inference is one observation at a time, so extra intra-op threads only add
    # contention. With the default thread count, 14 evals run in parallel pushed
    # the load average to ~80 on 16 cores.
    torch.set_num_threads(1)
    config = load_yaml_config(Path(args.config))
    env_cfg = dict(config["environment"])
    # The heuristic and the cloning agent decode the legacy observation vector,
    # which has a fixed number of sheep slots.
    if args.model_type in ("heuristic", "behavioral_cloning"):
        env_cfg["observation_mode"] = "legacy"
        env_cfg["randomize_sheep_count"] = False
    if args.fixed_sheep_count:
        env_cfg["randomize_sheep_count"] = False
    evaluation_cfg = config.get("evaluation", {})

    if args.scenarios:
        scenario_names = list(args.scenarios)
    else:
        scenario_names = list(
            evaluation_cfg.get("test_scenarios", ["test_procedural"])
        ) + list(evaluation_cfg.get("unseen_scenarios", []))
    scenarios = [
        ("test" if name.startswith("test") else "unseen", name) for name in scenario_names
    ]

    if args.model_type == "heuristic":
        model = HeuristicShepherdAgent(
            n_sheep=int(env_cfg["n_sheep"]),
            max_obstacles=int(env_cfg["max_obstacles"]),
            grid_size=float(env_cfg["grid_size"]),
            visibility_radius=float(env_cfg["visibility_radius"]),
            flee_radius=float(env_cfg["flee_radius"]),
            success_radius=float(env_cfg["success_radius"]),
            use_cluster_targets=bool(
                config.get("imitation", {}).get("expert", {}).get("use_cluster_targets", False)
            ),
        )
    elif args.model_path is None:
        raise SystemExit("--model-path is required for this model type.")
    else:
        model = load_model(args.model_type, args.model_path)
    obs_normalizer = (
        load_obs_normalizer(Path(args.vecnormalize)) if args.vecnormalize else None
    )
    if obs_normalizer is None and args.model_type in ("recurrent", "feedforward"):
        print(
            "[warning] No --vecnormalize given. If the model was trained with "
            "observation normalization enabled, these scores will be meaningless."
        )

    rows, summaries = evaluate_scenarios(
        env_config=env_cfg,
        scenarios=scenarios,
        model=model,
        model_type=args.model_type,
        run_name=args.run_name or Path(args.model_path or args.model_type).stem,
        episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=bool(evaluation_cfg.get("deterministic", True)),
        obs_normalizer=obs_normalizer,
    )

    output_dir = Path(args.output_dir)
    save_rows(output_dir / "trajectories.csv", rows)
    save_summaries(output_dir / "episode_summaries.csv", summaries)

    df = pd.read_csv(output_dir / "episode_summaries.csv")
    report = []
    for (split, scenario), group in df.groupby(["split", "scenario"]):
        fractions = group["fraction_at_goal"].to_numpy(dtype=float)
        low, high = bootstrap_ci(fractions)
        report.append(
            {
                "split": split,
                "scenario": scenario,
                "episodes": int(len(group)),
                "success_rate": float(group["success"].mean()),
                "fraction_at_goal": float(fractions.mean()),
                "fraction_at_goal_ci_low": low,
                "fraction_at_goal_ci_high": high,
                "best_fraction_at_goal": float(group["best_fraction_at_goal"].mean()),
                "mean_dist_to_goal": float(group["mean_dist_to_goal"].mean()),
                "mean_episode_length": float(group["episode_length"].mean()),
            }
        )
    report_df = pd.DataFrame(report).sort_values(["split", "scenario"])
    report_df.to_csv(output_dir / "generalization_report.csv", index=False)
    write_json(
        output_dir / "generalization_metadata.json",
        {
            "model_path": args.model_path,
            "model_type": args.model_type,
            "vecnormalize": args.vecnormalize,
            "episodes_per_scenario": args.episodes,
            "seed_start": args.seed_start,
            "scenarios": scenario_names,
        },
    )

    print(report_df.to_string(index=False))
    print(f"\nWrote {output_dir}/generalization_report.csv")


if __name__ == "__main__":
    main()
