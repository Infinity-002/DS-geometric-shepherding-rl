"""Training callbacks for research experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ResearchMetricsCallback(BaseCallback):
    """Record research-focused rolling episode metrics."""

    def __init__(self, log_freq: int = 4096, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self.successes: List[float] = []
        self.returns: List[float] = []
        self.lengths: List[float] = []
        self.visibilities: List[float] = []
        self.strays: List[float] = []
        self.collisions: List[float] = []
        self.collision_events: List[float] = []
        self.progress_rewards: List[float] = []
        self.zero_visibility_penalties: List[float] = []
        self.collision_penalties: List[float] = []

    def _on_step(self) -> bool:
        infos: Sequence[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.array([]))

        for idx, done in enumerate(dones):
            if not done or idx >= len(infos):
                continue
            info = infos[idx]
            self.successes.append(float(info.get("all_at_goal", False)))
            self.returns.append(float(info.get("episode_return", 0.0)))
            self.lengths.append(float(info.get("step", 0)))
            self.visibilities.append(float(info.get("avg_visibility_ratio", 0.0)))
            self.strays.append(float(info.get("stray_count", 0)))
            self.collisions.append(float(info.get("collision_count", 0)))
            self.collision_events.append(float(info.get("collision_event_count", 0)))
            self.progress_rewards.append(float(info.get("avg_reward_progress", 0.0)))
            self.zero_visibility_penalties.append(
                float(info.get("avg_reward_zero_visibility", 0.0))
            )
            self.collision_penalties.append(float(info.get("avg_reward_collision", 0.0)))

        if self.num_timesteps % self.log_freq == 0 and self.successes:
            self.logger.record("research/success_rate", rolling_mean(self.successes))
            self.logger.record("research/episode_return", rolling_mean(self.returns))
            self.logger.record("research/episode_length", rolling_mean(self.lengths))
            self.logger.record("research/visibility_ratio", rolling_mean(self.visibilities))
            self.logger.record("research/stray_count", rolling_mean(self.strays))
            self.logger.record("research/collision_count", rolling_mean(self.collisions))
            self.logger.record(
                "research/collision_event_count", rolling_mean(self.collision_events)
            )
            self.logger.record(
                "research/reward_progress", rolling_mean(self.progress_rewards)
            )
            self.logger.record(
                "research/reward_zero_visibility",
                rolling_mean(self.zero_visibility_penalties),
            )
            self.logger.record(
                "research/reward_collision",
                rolling_mean(self.collision_penalties),
            )
            if self.verbose:
                print(
                    f"[Step {self.num_timesteps:>8d}] "
                    f"SR={rolling_mean(self.successes):.3f} | "
                    f"Ret={rolling_mean(self.returns):.2f} | "
                    f"Vis={rolling_mean(self.visibilities):.2f} | "
                    f"Prog={rolling_mean(self.progress_rewards):.3f} | "
                    f"CollEvt={rolling_mean(self.collision_events):.2f}"
                )
        return True


class LinearCurriculumCallback(BaseCallback):
    """Increase environment difficulty linearly with timesteps."""

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps = max(int(total_timesteps), 1)

    def _on_step(self) -> bool:
        progress = min(float(self.num_timesteps) / float(self.total_timesteps), 1.0)
        _set_stage_on_envs(self.training_env.envs, progress)
        return True


class AdaptiveCurriculumCallback(BaseCallback):
    """Advance curriculum based on rolling task performance."""

    def __init__(
        self,
        stages: Sequence[Dict[str, float]],
        window: int = 50,
        warmup_episodes: int = 10,
        total_timesteps: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.stages = sorted((dict(stage) for stage in stages), key=lambda item: item["stage"])
        self.window = max(int(window), 1)
        self.warmup_episodes = max(int(warmup_episodes), 1)
        self.total_timesteps = None if total_timesteps is None else max(int(total_timesteps), 1)
        self.successes: List[float] = []
        self.visibilities: List[float] = []
        self.collision_events: List[float] = []
        self.progress_rewards: List[float] = []
        self.current_stage = float(self.stages[0]["stage"]) if self.stages else 0.0

    def _on_training_start(self) -> None:
        _set_stage_on_envs(self.training_env.envs, self.current_stage)

    def _on_step(self) -> bool:
        infos: Sequence[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.array([]))
        for idx, done in enumerate(dones):
            if not done or idx >= len(infos):
                continue
            info = infos[idx]
            self.successes.append(float(info.get("all_at_goal", False)))
            self.visibilities.append(float(info.get("avg_visibility_ratio", 0.0)))
            self.collision_events.append(float(info.get("collision_event_count", 0.0)))
            self.progress_rewards.append(float(info.get("avg_reward_progress", 0.0)))

        next_stage = self._compute_stage()
        if next_stage != self.current_stage:
            self.current_stage = next_stage
            _set_stage_on_envs(self.training_env.envs, self.current_stage)
            if self.verbose:
                print(f"Adaptive curriculum advanced to stage {self.current_stage:.2f}")
        return True

    def _compute_stage(self) -> float:
        if not self.stages:
            return 0.0

        candidate = float(self.stages[0]["stage"])
        if len(self.successes) < self.warmup_episodes:
            return candidate

        success_rate = rolling_mean(self.successes, self.window)
        visibility_ratio = rolling_mean(self.visibilities, self.window)
        collision_event_count = rolling_mean(self.collision_events, self.window)
        progress_reward = rolling_mean(self.progress_rewards, self.window)

        for stage in self.stages:
            if success_rate < float(stage.get("min_success_rate", 0.0)):
                break
            if visibility_ratio < float(stage.get("min_visibility_ratio", 0.0)):
                break
            if collision_event_count > float(stage.get("max_collision_event_count", np.inf)):
                break
            if progress_reward < float(stage.get("min_progress_reward", -np.inf)):
                break
            if self.total_timesteps is not None:
                min_timestep_ratio = float(stage.get("min_timestep_ratio", 0.0))
                current_ratio = float(self.num_timesteps) / float(self.total_timesteps)
                if current_ratio < min_timestep_ratio:
                    break
            candidate = float(stage["stage"])
        return candidate


def build_curriculum_callback(
    total_timesteps: int,
    curriculum_cfg: Dict[str, Any] | None,
    verbose: int = 0,
) -> BaseCallback:
    curriculum_cfg = curriculum_cfg or {}
    strategy = str(curriculum_cfg.get("strategy", "adaptive")).lower()
    if strategy == "linear":
        return LinearCurriculumCallback(total_timesteps=total_timesteps, verbose=verbose)
    stages = curriculum_cfg.get("stages", [{"stage": 0.0}, {"stage": 1.0}])
    return AdaptiveCurriculumCallback(
        stages=stages,
        window=int(curriculum_cfg.get("window", 50)),
        warmup_episodes=int(curriculum_cfg.get("warmup_episodes", 10)),
        total_timesteps=total_timesteps,
        verbose=verbose,
    )


def rolling_mean(values: Sequence[float], window: int = 100) -> float:
    arr = np.asarray(values[-window:], dtype=np.float32)
    return float(arr.mean()) if arr.size else 0.0


def _set_stage_on_envs(envs: Sequence[Any], stage: float) -> None:
    for env in envs:
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        if hasattr(inner, "set_curriculum_stage"):
            inner.set_curriculum_stage(stage)
