"""Environment and model construction helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from shepherding.imitation import load_behavioral_cloning_agent

import shepherding.envs  # noqa: F401


VECNORMALIZE_FILENAME = "vecnormalize.pkl"


def make_research_env(env_config: Dict[str, Any], seed: int, scenario: str) -> gym.Env:
    """Build a single, unvectorized environment (evaluation and rendering)."""
    cfg = dict(env_config)
    cfg.pop("n_envs", None)
    cfg.pop("vec_env_type", None)
    cfg.pop("normalize_observations", None)
    cfg.pop("normalize_rewards", None)
    cfg["scenario"] = scenario
    env = gym.make("HerdingEnv-v3", **cfg)
    env = Monitor(env)
    env.reset(seed=seed, options={"scenario": scenario})
    return env


def _env_factory(
    env_config: Dict[str, Any], seed: int, scenario: str, rank: int
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        cfg = dict(env_config)
        cfg["scenario"] = scenario
        env = gym.make("HerdingEnv-v3", **cfg)
        env.reset(seed=seed + rank, options={"scenario": scenario})
        return env

    return _init


def make_research_vec_env(
    env_config: Dict[str, Any],
    seed: int,
    scenario: str,
    n_envs: int = 8,
    vec_env_type: str = "subproc",
    normalize_observations: bool = True,
    normalize_rewards: bool = True,
    clip_obs: float = 10.0,
    gamma: float = 0.99,
    start_method: Optional[str] = None,
) -> VecEnv:
    """Build the vectorized, normalized training environment.

    Training on a single environment means every PPO batch comes from one
    domain-randomized scenario, so the policy chases whichever layout it happens
    to be in. Running many workers puts independent randomization draws in the
    same gradient batch. ``VecNormalize`` then keeps observation and return scales
    stable, which matters because the terminal success bonus is orders of
    magnitude larger than the per-step shaping terms.
    """
    cfg = dict(env_config)
    for key in ("n_envs", "vec_env_type", "normalize_observations", "normalize_rewards"):
        cfg.pop(key, None)

    n_envs = max(int(n_envs), 1)
    factories = [_env_factory(cfg, seed, scenario, rank) for rank in range(n_envs)]

    if str(vec_env_type).lower() == "subproc" and n_envs > 1:
        # start_method=None lets SB3 pick the platform default (forkserver on
        # Linux), which starts faster than spawn. Any subprocess start method
        # re-imports the entry point, so callers must be importable modules or
        # scripts guarded by ``if __name__ == "__main__":`` -- use
        # vec_env_type="dummy" from an interactive session or a stdin heredoc.
        vec_env: VecEnv = SubprocVecEnv(factories, start_method=start_method)
    else:
        vec_env = DummyVecEnv(factories)

    vec_env = VecMonitor(vec_env)
    if normalize_observations or normalize_rewards:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=bool(normalize_observations),
            norm_reward=bool(normalize_rewards),
            clip_obs=float(clip_obs),
            gamma=float(gamma),
        )
    vec_env.seed(seed)
    return vec_env


def save_vecnormalize(env: Any, directory: Path, filename: str = VECNORMALIZE_FILENAME) -> Optional[Path]:
    """Persist ``VecNormalize`` statistics next to the model, if present."""
    normalizer = _find_vecnormalize(env)
    if normalizer is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    normalizer.save(str(path))
    return path


def load_vecnormalize(env: VecEnv, path: Path) -> VecEnv:
    """Reattach saved normalization statistics for evaluation (frozen)."""
    normalizer = VecNormalize.load(str(path), env)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def load_obs_normalizer(path: Path) -> Callable[[Any], Any]:
    """Return a function applying saved ``VecNormalize`` observation statistics.

    Evaluation runs a single un-vectorized environment so that ``env.unwrapped``
    stays reachable for trajectory logging. This keeps that loop intact while
    still feeding the policy observations on the scale it was trained on — a
    model trained under ``VecNormalize`` scores near-randomly on raw observations.
    """
    import pickle

    with open(path, "rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False

    def _normalize(observation: Any) -> Any:
        return normalizer.normalize_obs(observation)

    return _normalize


def _find_vecnormalize(env: Any) -> Optional[VecNormalize]:
    current = env
    while current is not None:
        if isinstance(current, VecNormalize):
            return current
        current = getattr(current, "venv", None)
    return None


def _with_lr_schedule(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``lr_schedule: linear`` into a callable SB3 learning rate.

    Linear decays ``learning_rate`` to zero over the run, as in the original
    PPO paper. Any other value (or none) keeps the rate constant.
    """
    schedule = str(config.pop("lr_schedule", "constant")).lower()
    if schedule == "linear":
        initial = float(config.get("learning_rate", 3e-4))
        config["learning_rate"] = lambda progress_remaining: initial * progress_remaining
    return config


def build_feedforward_model(
    env: gym.Env,
    ppo_config: Dict[str, Any],
    seed: int,
    tensorboard_log: str | None,
) -> PPO:
    tensorboard_log = _maybe_disable_tensorboard(tensorboard_log)
    config = _with_lr_schedule(dict(ppo_config))
    policy_kwargs = config.pop("policy_kwargs", None)
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        **config,
    )


def build_recurrent_model(
    env: gym.Env,
    ppo_config: Dict[str, Any],
    seed: int,
    tensorboard_log: str | None,
) -> RecurrentPPO:
    model_config = _with_lr_schedule(dict(ppo_config))
    lstm_hidden_size = int(model_config.pop("lstm_hidden_size", 256))
    policy_kwargs = dict(model_config.pop("policy_kwargs", None) or {})
    policy_kwargs.setdefault("lstm_hidden_size", lstm_hidden_size)
    tensorboard_log = _maybe_disable_tensorboard(tensorboard_log)
    return RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
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
