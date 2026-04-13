"""Evaluation and aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from shepherding.research.models import make_research_env


@dataclass
class EpisodeSummary:
    run_name: str
    model_type: str
    seed: int
    split: str
    scenario: str
    episode_idx: int
    success: int
    episode_length: int
    episode_return: float
    mean_dist_to_goal: float
    visible_ratio: float
    avg_visibility_ratio: float
    flock_hull_area: float
    stray_count: int
    collision_count: int
    collision_event_count: int
    dog_path_length: float
    avg_reward_base: float
    avg_reward_progress: float
    avg_reward_worst_sheep: float
    avg_reward_visibility_loss: float
    avg_reward_visibility_gain: float
    avg_reward_zero_visibility: float
    avg_reward_stray: float
    avg_reward_drive: float
    avg_reward_collision: float
    avg_reward_success_bonus: float


def collect_episode(
    env: Any,
    model: Any,
    deterministic: bool,
    max_steps: int,
    run_name: str,
    model_type: str,
    seed: int,
    split: str,
    scenario: str,
    episode_idx: int,
) -> tuple[list[dict[str, Any]], EpisodeSummary]:
    """Run a single episode and return step-wise rows and final summary."""
    obs, info = env.reset(seed=seed, options={"scenario": scenario})
    rows: List[Dict[str, Any]] = []
    final_info = info

    if hasattr(model, "reset"):
        model.reset()

    state = None
    episode_start = np.ones((1,), dtype=bool)
    total_reward = 0.0

    for step in range(max_steps):
        action, state = _predict_action(
            model=model,
            model_type=model_type,
            observation=obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            env=env,
        )

        next_obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated

        inner = env.unwrapped
        centroid = np.mean(inner.sheep_pos, axis=0)
        rows.append(
            {
                "run_name": run_name,
                "model_type": model_type,
                "seed": seed,
                "split": split,
                "scenario": scenario,
                "episode_idx": episode_idx,
                "step": step,
                "reward": float(reward),
                "done": int(done),
                "dog_x": float(inner.dog_pos[0]),
                "dog_y": float(inner.dog_pos[1]),
                "goal_x": float(inner.goal[0]),
                "goal_y": float(inner.goal[1]),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "mean_dist_to_goal": float(final_info.get("mean_dist_to_goal", 0.0)),
                "visible_ratio": float(final_info.get("visible_ratio", 0.0)),
                "flock_hull_area": float(final_info.get("flock_hull_area", 0.0)),
                "stray_count": int(final_info.get("stray_count", 0)),
                "collision_count": int(final_info.get("collision_count", 0)),
                "collision_event_count": int(final_info.get("collision_event_count", 0)),
                "progress_to_goal": float(final_info.get("progress_to_goal", 0.0)),
                "reward_base": float(final_info.get("reward_base", 0.0)),
                "reward_progress": float(final_info.get("reward_progress", 0.0)),
                "reward_visibility_loss": float(
                    final_info.get("reward_visibility_loss", 0.0)
                ),
                "reward_visibility_gain": float(
                    final_info.get("reward_visibility_gain", 0.0)
                ),
                "reward_zero_visibility": float(
                    final_info.get("reward_zero_visibility", 0.0)
                ),
                "reward_stray": float(final_info.get("reward_stray", 0.0)),
                "reward_drive": float(final_info.get("reward_drive", 0.0)),
                "reward_collision": float(final_info.get("reward_collision", 0.0)),
                "reward_success_bonus": float(
                    final_info.get("reward_success_bonus", 0.0)
                ),
            }
        )

        obs = next_obs
        episode_start = np.array([done], dtype=bool)
        if done:
            break

    summary = EpisodeSummary(
        run_name=run_name,
        model_type=model_type,
        seed=seed,
        split=split,
        scenario=scenario,
        episode_idx=episode_idx,
        success=int(final_info.get("all_at_goal", False)),
        episode_length=int(final_info.get("step", len(rows))),
        episode_return=float(total_reward),
        mean_dist_to_goal=float(final_info.get("mean_dist_to_goal", 0.0)),
        visible_ratio=float(final_info.get("visible_ratio", 0.0)),
        avg_visibility_ratio=float(final_info.get("avg_visibility_ratio", 0.0)),
        flock_hull_area=float(final_info.get("flock_hull_area", 0.0)),
        stray_count=int(final_info.get("stray_count", 0)),
        collision_count=int(final_info.get("collision_count", 0)),
        collision_event_count=int(final_info.get("collision_event_count", 0)),
        dog_path_length=float(final_info.get("dog_path_length", 0.0)),
        avg_reward_base=float(final_info.get("avg_reward_base", 0.0)),
        avg_reward_progress=float(final_info.get("avg_reward_progress", 0.0)),
        avg_reward_worst_sheep=float(final_info.get("avg_reward_worst_sheep", 0.0)),
        avg_reward_visibility_loss=float(
            final_info.get("avg_reward_visibility_loss", 0.0)
        ),
        avg_reward_visibility_gain=float(
            final_info.get("avg_reward_visibility_gain", 0.0)
        ),
        avg_reward_zero_visibility=float(
            final_info.get("avg_reward_zero_visibility", 0.0)
        ),
        avg_reward_stray=float(final_info.get("avg_reward_stray", 0.0)),
        avg_reward_drive=float(final_info.get("avg_reward_drive", 0.0)),
        avg_reward_collision=float(final_info.get("avg_reward_collision", 0.0)),
        avg_reward_success_bonus=float(
            final_info.get("avg_reward_success_bonus", 0.0)
        ),
    )
    return rows, summary


def evaluate_scenarios(
    env_config: Dict[str, Any],
    scenarios: Sequence[Tuple[str, str]],
    model: Any,
    model_type: str,
    run_name: str,
    episodes: int,
    seed_start: int,
    deterministic: bool,
) -> tuple[list[dict[str, Any]], list[EpisodeSummary]]:
    rows: list[dict[str, Any]] = []
    summaries: list[EpisodeSummary] = []

    eval_env_cfg = dict(env_config)
    eval_env_cfg["compute_expensive_metrics"] = True
    for split, scenario in scenarios:
        env = make_research_env(eval_env_cfg, seed=seed_start, scenario=scenario)
        for episode_idx in range(episodes):
            episode_seed = seed_start + episode_idx
            episode_rows, summary = collect_episode(
                env=env,
                model=model,
                deterministic=deterministic,
                max_steps=int(eval_env_cfg["max_steps"]),
                run_name=run_name,
                model_type=model_type,
                seed=episode_seed,
                split=split,
                scenario=scenario,
                episode_idx=episode_idx,
            )
            rows.extend(episode_rows)
            summaries.append(summary)
        env.close()
    return rows, summaries


def aggregate_results(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    agg_spec: Dict[str, tuple[str, str]] = {
        "success_rate": ("success", "mean"),
        "mean_episode_return": ("episode_return", "mean"),
        "std_episode_return": ("episode_return", "std"),
        "mean_episode_length": ("episode_length", "mean"),
        "mean_dist_to_goal": ("mean_dist_to_goal", "mean"),
        "mean_visible_ratio": ("visible_ratio", "mean"),
        "mean_avg_visibility": ("avg_visibility_ratio", "mean"),
        "mean_hull_area": ("flock_hull_area", "mean"),
        "mean_stray_count": ("stray_count", "mean"),
        "mean_collision_count": ("collision_count", "mean"),
        "mean_dog_path_length": ("dog_path_length", "mean"),
        "seeds": ("seed", "nunique"),
    }
    optional_aggs: Dict[str, tuple[str, str]] = {
        "mean_collision_event_count": ("collision_event_count", "mean"),
        "mean_reward_base": ("avg_reward_base", "mean"),
        "mean_reward_progress": ("avg_reward_progress", "mean"),
        "mean_reward_worst_sheep": ("avg_reward_worst_sheep", "mean"),
        "mean_reward_visibility_loss": ("avg_reward_visibility_loss", "mean"),
        "mean_reward_visibility_gain": ("avg_reward_visibility_gain", "mean"),
        "mean_reward_zero_visibility": ("avg_reward_zero_visibility", "mean"),
        "mean_reward_stray": ("avg_reward_stray", "mean"),
        "mean_reward_drive": ("avg_reward_drive", "mean"),
        "mean_reward_collision": ("avg_reward_collision", "mean"),
        "mean_reward_success_bonus": ("avg_reward_success_bonus", "mean"),
    }
    for out_col, (in_col, op) in optional_aggs.items():
        if in_col in df.columns:
            agg_spec[out_col] = (in_col, op)

    grouped = (
        df.groupby(["run_name", "model_type", "split", "scenario"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["split", "scenario", "model_type"])
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    return grouped


def create_significance_table(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    comparisons: List[Dict[str, Any]] = []
    grouped = df.groupby(["split", "scenario"])
    for (split, scenario), group in grouped:
        run_names = sorted(group["run_name"].unique().tolist())
        for first_idx, first_name in enumerate(run_names):
            for second_name in run_names[first_idx + 1 :]:
                first = group[group["run_name"] == first_name]["episode_return"].to_numpy()
                second = group[group["run_name"] == second_name]["episode_return"].to_numpy()
                if len(first) == 0 or len(second) == 0:
                    continue
                stat = ttest_ind(first, second, equal_var=False)
                comparisons.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "run_a": first_name,
                        "run_b": second_name,
                        "run_a_mean_return": float(np.mean(first)),
                        "run_b_mean_return": float(np.mean(second)),
                        "return_gap": float(np.mean(first) - np.mean(second)),
                        "p_value": float(stat.pvalue),
                    }
                )
    out_df = pd.DataFrame(comparisons)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def _predict_action(
    model: Any,
    model_type: str,
    observation: np.ndarray,
    state: Any,
    episode_start: np.ndarray,
    deterministic: bool,
    env: Any,
) -> tuple[np.ndarray, Any]:
    if model is None:
        return env.action_space.sample(), state
    if model_type == "recurrent":
        action, next_state = model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return np.asarray(action, dtype=np.float32), next_state
    prediction = model.predict(observation, deterministic=deterministic)
    if isinstance(prediction, tuple):
        action = prediction[0]
    else:
        action = prediction
    return np.asarray(action, dtype=np.float32), state
