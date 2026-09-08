"""Shared training routine for the v3 research environment.

Both entry-point scripts (feedforward and recurrent) run the same pipeline:
vectorized + normalized training envs, a curriculum callback whose stage
occupancy is recorded, periodic evaluation on a held-out validation
distribution for model selection, and a metadata dump that makes the run
diagnosable after the fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from shepherding.research.callbacks import (
    GeneralizationEvalCallback,
    ResearchMetricsCallback,
    build_curriculum_callback,
    collect_stage_summary,
)
from shepherding.research.io import write_json
from shepherding.research.models import (
    VECNORMALIZE_FILENAME,
    build_feedforward_model,
    build_recurrent_model,
    make_research_vec_env,
    save_vecnormalize,
)


def train_v3(
    config: Dict[str, Any],
    model_type: str,
    seed: int,
    total_timesteps: Optional[int] = None,
    run_name: Optional[str] = None,
    scenario: str = "train",
    config_path: str = "",
) -> Path:
    """Train one v3 agent and return the path to the saved model directory."""
    if model_type not in ("feedforward", "recurrent"):
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    env_cfg = dict(config["environment"])
    train_cfg = config["training"]
    ppo_cfg = dict(config[f"ppo_{model_type}"])
    vec_cfg = dict(train_cfg.get("vec_env", {}))

    total_timesteps = int(total_timesteps or train_cfg["total_timesteps"])
    run_name = run_name or f"{model_type}_seed{seed}"

    n_envs = int(vec_cfg.get("n_envs", 8))
    env = make_research_vec_env(
        env_cfg,
        seed=seed,
        scenario=scenario,
        n_envs=n_envs,
        vec_env_type=str(vec_cfg.get("vec_env_type", "subproc")),
        normalize_observations=bool(vec_cfg.get("normalize_observations", True)),
        normalize_rewards=bool(vec_cfg.get("normalize_rewards", True)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        start_method=vec_cfg.get("start_method"),
    )

    tensorboard_log = f"{train_cfg['tensorboard_log']}/{model_type}"
    builder = build_feedforward_model if model_type == "feedforward" else build_recurrent_model
    model = builder(
        env=env,
        ppo_config=ppo_cfg,
        seed=seed,
        tensorboard_log=tensorboard_log,
    )

    save_dir = Path(train_cfg["save_dir"]) / model_type
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_callback = ResearchMetricsCallback(log_freq=2048, verbose=1)
    curriculum_callback = build_curriculum_callback(
        total_timesteps, train_cfg.get("curriculum"), verbose=1
    )
    callbacks: List[Any] = [metrics_callback, curriculum_callback]

    validation_cfg = dict(train_cfg.get("validation", {}))
    eval_callback: Optional[GeneralizationEvalCallback] = None
    if validation_cfg.get("enabled", True):
        eval_callback = GeneralizationEvalCallback(
            env_config=env_cfg,
            model_type=model_type,
            scenario=str(validation_cfg.get("scenario", "validation")),
            n_episodes=int(validation_cfg.get("episodes", 12)),
            eval_freq=int(validation_cfg.get("eval_freq", 25_000)),
            seed=int(validation_cfg.get("seed", 10_000)) + seed,
            best_model_path=save_dir / f"{run_name}_best",
            training_env=env,
            verbose=1,
        )
        callbacks.append(eval_callback)

    print("=" * 72)
    print(f"  Shepherding RL v3 – {model_type.capitalize()} PPO")
    print(f"  Run name         : {run_name}")
    print(f"  Seed             : {seed}")
    print(f"  Scenario         : {scenario}")
    print(f"  Parallel envs    : {n_envs}")
    print(f"  Total timesteps  : {total_timesteps:,}")
    print("=" * 72)

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    model_path = save_dir / run_name
    model.save(str(model_path))
    normalizer_path = save_vecnormalize(env, save_dir, f"{run_name}_{VECNORMALIZE_FILENAME}")

    metadata: Dict[str, Any] = {
        "run_name": run_name,
        "model_type": model_type,
        "seed": seed,
        "scenario": scenario,
        "total_timesteps": total_timesteps,
        "config_path": config_path,
        "n_envs": n_envs,
        "observation_mode": env_cfg.get("observation_mode", "egocentric"),
        "observation_frame": env_cfg.get("observation_frame", "world"),
        "vecnormalize_path": str(normalizer_path) if normalizer_path else None,
        "curriculum": collect_stage_summary(callbacks),
    }
    if eval_callback is not None:
        metadata["validation"] = eval_callback.summary()
    write_json(save_dir / f"{run_name}_metadata.json", metadata)

    curriculum = metadata["curriculum"] or {}
    if curriculum and not curriculum.get("reached_final_stage", False):
        print(
            "\n[warning] The curriculum finished at stage "
            f"{curriculum.get('final_stage', 0.0):.2f} (< 1.0). Earlier stages narrow "
            "the obstacle topologies and shrink the dynamics ranges, so this run "
            "trained on less randomization than the config asks for. Check "
            "curriculum.fraction_per_stage in this metadata file, then loosen "
            "training.curriculum.stages or train for longer."
        )
    print(f"\nSaved model to {model_path}.zip")
    env.close()
    return model_path
