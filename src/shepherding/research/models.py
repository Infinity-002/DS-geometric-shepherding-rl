"""Environment and model construction helpers."""

from __future__ import annotations

import importlib.util
from typing import Any, Dict

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from shepherding.imitation import load_behavioral_cloning_agent

import shepherding.envs  # noqa: F401


def make_research_env(env_config: Dict[str, Any], seed: int, scenario: str) -> gym.Env:
    cfg = dict(env_config)
    cfg["scenario"] = scenario
    env = gym.make("HerdingEnv-v3", **cfg)
    env = Monitor(env)
    env.reset(seed=seed, options={"scenario": scenario})
    return env


def build_feedforward_model(
    env: gym.Env,
    ppo_config: Dict[str, Any],
    seed: int,
    tensorboard_log: str | None,
) -> PPO:
    tensorboard_log = _maybe_disable_tensorboard(tensorboard_log)
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        **ppo_config,
    )


def build_recurrent_model(
    env: gym.Env,
    ppo_config: Dict[str, Any],
    seed: int,
    tensorboard_log: str | None,
) -> RecurrentPPO:
    model_config = dict(ppo_config)
    lstm_hidden_size = int(model_config.pop("lstm_hidden_size", 256))
    tensorboard_log = _maybe_disable_tensorboard(tensorboard_log)
    return RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs={"lstm_hidden_size": lstm_hidden_size},
        **model_config,
    )


def load_model(model_type: str, model_path: str) -> Any:
    if model_type == "recurrent":
        return RecurrentPPO.load(model_path)
    if model_type == "feedforward":
        return PPO.load(model_path)
    if model_type == "behavioral_cloning":
        return load_behavioral_cloning_agent(model_path)
    raise ValueError(f"Unsupported model_type '{model_type}' for load_model().")


def _maybe_disable_tensorboard(tensorboard_log: str | None) -> str | None:
    if tensorboard_log is None:
        return None
    if importlib.util.find_spec("tensorboard") is not None:
        return tensorboard_log
    print("TensorBoard is not installed; continuing without tensorboard logging.")
    return None
