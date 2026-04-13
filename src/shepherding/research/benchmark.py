"""Benchmark orchestration for research experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from stable_baselines3.common.callbacks import BaseCallback

from shepherding.baselines import HeuristicShepherdAgent
from shepherding.imitation import load_behavioral_cloning_agent
from shepherding.research.callbacks import ResearchMetricsCallback, build_curriculum_callback
from shepherding.research.evaluation import EpisodeSummary, evaluate_scenarios
from shepherding.research.io import save_rows, save_summaries, write_json
from shepherding.research.models import (
    build_feedforward_model,
    build_recurrent_model,
    make_research_env,
)


def run_benchmark(
    config: Dict[str, Any],
    benchmark_cfg: Dict[str, Any],
    output_dir: Path,
    save_models: bool = False,
) -> tuple[list[dict[str, Any]], list[EpisodeSummary]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = config["environment"]
    training_cfg = config["training"]
    evaluation_cfg = config["evaluation"]
    total_timesteps = int(benchmark_cfg.get("total_timesteps", training_cfg["total_timesteps"]))
    episodes = int(benchmark_cfg.get("episodes", training_cfg["eval_episodes"]))
    seeds = list(benchmark_cfg.get("seeds", [0]))
    scenarios = _benchmark_scenarios(evaluation_cfg, benchmark_cfg)
    variants = list(benchmark_cfg.get("variants", []))

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[EpisodeSummary] = []

    for variant in variants:
        for seed in seeds:
            run_name = f"{variant['run_name']}_seed{seed}"
            env_config = deepcopy(env_cfg)
            env_config.update(variant.get("env_overrides", {}))
            model_type = str(variant["model_type"])
            model = _build_or_train_model(
                run_name=run_name,
                model_type=model_type,
                env_config=env_config,
                config=config,
                total_timesteps=total_timesteps,
                seed=seed,
                save_models=save_models,
                output_dir=output_dir,
            )
            rows, summaries = evaluate_scenarios(
                env_config=env_config,
                scenarios=scenarios,
                model=model,
                model_type=model_type,
                run_name=variant["run_name"],
                episodes=episodes,
                seed_start=seed,
                deterministic=bool(evaluation_cfg.get("deterministic", True)),
            )
            all_rows.extend(rows)
            all_summaries.extend(summaries)

    save_rows(output_dir / "trajectories.csv", all_rows)
    save_summaries(output_dir / "episode_summaries.csv", all_summaries)
    write_json(
        output_dir / "benchmark_metadata.json",
        {
            "seeds": seeds,
            "episodes_per_scenario": episodes,
            "variants": [variant["run_name"] for variant in variants],
            "scenarios": [{"split": split, "scenario": scenario} for split, scenario in scenarios],
        },
    )
    return all_rows, all_summaries


def benchmark_callbacks(training_cfg: Dict[str, Any], total_timesteps: int) -> list[BaseCallback]:
    callbacks: list[BaseCallback] = [ResearchMetricsCallback(log_freq=2048, verbose=1)]
    curriculum_cfg = training_cfg.get("curriculum", {})
    callbacks.append(build_curriculum_callback(total_timesteps, curriculum_cfg, verbose=1))
    return callbacks


def _build_or_train_model(
    run_name: str,
    model_type: str,
    env_config: Dict[str, Any],
    config: Dict[str, Any],
    total_timesteps: int,
    seed: int,
    save_models: bool,
    output_dir: Path,
) -> Any:
    training_cfg = config["training"]
    if model_type == "heuristic":
        return HeuristicShepherdAgent(
            n_sheep=int(env_config["n_sheep"]),
            max_obstacles=int(env_config["max_obstacles"]),
            grid_size=float(env_config["grid_size"]),
            visibility_radius=float(env_config["visibility_radius"]),
            flee_radius=float(env_config["flee_radius"]),
            success_radius=float(env_config["success_radius"]),
            use_cluster_targets=bool(config.get("imitation", {}).get("expert", {}).get("use_cluster_targets", False)),
        )
    if model_type == "behavioral_cloning":
        model_path = str(config["imitation"]["training"]["model_path"])
        return load_behavioral_cloning_agent(model_path)

    train_env = make_research_env(env_config, seed=seed, scenario="train")
    callbacks = benchmark_callbacks(training_cfg, total_timesteps)
    tensorboard_root = str(training_cfg.get("tensorboard_log", "runs/research_v3"))

    if model_type == "recurrent":
        model = build_recurrent_model(
            env=train_env,
            ppo_config=config["ppo_recurrent"],
            seed=seed,
            tensorboard_log=f"{tensorboard_root}/{run_name}",
        )
    elif model_type == "feedforward":
        model = build_feedforward_model(
            env=train_env,
            ppo_config=config["ppo_feedforward"],
            seed=seed,
            tensorboard_log=f"{tensorboard_root}/{run_name}",
        )
    else:
        train_env.close()
        raise ValueError(f"Unknown benchmark model_type '{model_type}'.")

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    train_env.close()

    if save_models:
        model_dir = output_dir / "models" / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir / run_name))
    return model


def _benchmark_scenarios(
    evaluation_cfg: Dict[str, Any],
    benchmark_cfg: Dict[str, Any],
) -> Sequence[Tuple[str, str]]:
    scenario_cfg = benchmark_cfg.get("scenarios", {})
    train_names = list(scenario_cfg.get("train", evaluation_cfg.get("train_scenarios", ["train"])))
    unseen_names = list(
        scenario_cfg.get("unseen", evaluation_cfg.get("unseen_scenarios", []))
    )
    return [("train", name) for name in train_names] + [
        ("unseen", name) for name in unseen_names
    ]
