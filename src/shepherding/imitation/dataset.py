"""Demonstration dataset generation for imitation learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from shepherding.baselines import HeuristicShepherdAgent
from shepherding.imitation.features import feature_names, observation_to_features
from shepherding.research.io import save_rows, write_json
from shepherding.research.models import make_research_env


def collect_demonstrations(
    env_config: Dict[str, Any],
    collection_config: Dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Roll out a heuristic policy and save state-action demonstrations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _scenario_schedule(
        collection_config.get("train_scenarios", ["train"]),
        collection_config.get("unseen_scenarios", []),
    )
    target_steps = int(collection_config.get("target_steps", 10000))
    max_episodes = int(collection_config.get("max_episodes", 256))
    seed = int(collection_config.get("seed", 0))
    deterministic = bool(collection_config.get("deterministic", True))

    expert = HeuristicShepherdAgent(
        n_sheep=int(env_config["n_sheep"]),
        max_obstacles=int(env_config["max_obstacles"]),
        grid_size=float(env_config["grid_size"]),
        visibility_radius=float(env_config["visibility_radius"]),
        flee_radius=float(env_config["flee_radius"]),
        success_radius=float(env_config["success_radius"]),
        use_cluster_targets=bool(collection_config.get("use_cluster_targets", True)),
        cluster_activation_distance=float(
            collection_config.get("cluster_activation_distance", 2.0)
        ),
    )

    env_cfg = dict(env_config)
    env_cfg["compute_expensive_metrics"] = True
    rows: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    total_steps = 0
    episodes_run = 0

    feature_columns = feature_names(
        int(env_config["n_sheep"]),
        int(env_config["max_obstacles"]),
    )

    while total_steps < target_steps and episodes_run < max_episodes:
        split, scenario = scenarios[episodes_run % len(scenarios)]
        episode_seed = seed + episodes_run
        env = make_research_env(env_cfg, seed=episode_seed, scenario=scenario)
        obs, info = env.reset(seed=episode_seed, options={"scenario": scenario})
        expert.reset()

        episode_steps = 0
        episode_return = 0.0
        done = False
        episode_start = np.ones((1,), dtype=bool)

        while not done and total_steps < target_steps:
            action, _ = expert.predict(
                obs,
                state=None,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            feature_vector = observation_to_features(obs, int(env_config["n_sheep"]))
            next_obs, reward, terminated, truncated, info = env.step(action)

            row = {
                "episode_id": episodes_run,
                "split": split,
                "scenario": scenario,
                "seed": episode_seed,
                "step": episode_steps,
                "target_dx": float(action[0]),
                "target_dy": float(action[1]),
                "reward": float(reward),
            }
            for idx, value in enumerate(np.asarray(obs, dtype=np.float32).flatten()):
                row[f"obs_{idx}"] = float(value)
            for name, value in zip(feature_columns, feature_vector):
                row[name] = float(value)
            rows.append(row)

            total_steps += 1
            episode_steps += 1
            episode_return += float(reward)
            obs = next_obs
            done = bool(terminated or truncated)
            episode_start = np.array([done], dtype=bool)

        episode_summaries.append(
            {
                "episode_id": episodes_run,
                "split": split,
                "scenario": scenario,
                "seed": episode_seed,
                "steps": episode_steps,
                "episode_return": episode_return,
                "success": int(info.get("all_at_goal", False)),
            }
        )
        env.close()
        episodes_run += 1

    save_rows(output_dir / "demonstrations.csv", rows)
    save_rows(output_dir / "episodes.csv", episode_summaries)
    metadata = {
        "target_steps": target_steps,
        "collected_steps": total_steps,
        "episodes_run": episodes_run,
        "feature_names": feature_columns,
        "observation_dim": 4 + (2 * int(env_config["n_sheep"])) + (4 * int(env_config["max_obstacles"])),
        "expert_policy": "cluster_aware_heuristic"
        if collection_config.get("use_cluster_targets", True)
        else "heuristic",
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def _scenario_schedule(
    train_scenarios: Sequence[str],
    unseen_scenarios: Sequence[str],
) -> list[tuple[str, str]]:
    scheduled = [("train", scenario) for scenario in train_scenarios]
    scheduled.extend(("unseen", scenario) for scenario in unseen_scenarios)
    return scheduled or [("train", "train")]
