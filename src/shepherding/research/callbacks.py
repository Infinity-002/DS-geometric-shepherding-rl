"""Training callbacks for research experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        self.fractions_at_goal: List[float] = []
        self.curriculum_stages: List[float] = []

    def _on_step(self) -> bool:
        infos: Sequence[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.array([]))

        for idx, done in enumerate(dones):
            if not done or idx >= len(infos):
                continue
            info = infos[idx]
            self.fractions_at_goal.append(float(info.get("fraction_at_goal", 0.0)))
            self.curriculum_stages.append(float(info.get("curriculum_stage", 0.0)))
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
            self.logger.record(
                "research/fraction_at_goal", rolling_mean(self.fractions_at_goal)
            )
            self.logger.record(
                "research/curriculum_stage", rolling_mean(self.curriculum_stages, window=10)
            )
            if self.verbose:
                print(
                    f"[Step {self.num_timesteps:>8d}] "
                    f"SR={rolling_mean(self.successes):.3f} | "
                    f"FracGoal={rolling_mean(self.fractions_at_goal):.3f} | "
                    f"Ret={rolling_mean(self.returns):.2f} | "
                    f"Vis={rolling_mean(self.visibilities):.2f} | "
                    f"Prog={rolling_mean(self.progress_rewards):.3f} | "
                    f"CollEvt={rolling_mean(self.collision_events):.2f} | "
                    f"Stage={rolling_mean(self.curriculum_stages, window=10):.2f}"
                )
        return True


class StageTracker:
    """Accumulate how many timesteps were spent at each curriculum stage.

    Item 5 of the generalization plan: if the adaptive curriculum never reaches
    stage 1.0, the agent never trains on randomized obstacles/dynamics, which by
    itself explains a large seen/unseen gap. This makes that visible instead of
    something you have to infer from final scores.
    """

    def __init__(self) -> None:
        self.timesteps_per_stage: Dict[float, int] = {}
        self.stage_transitions: List[Dict[str, float]] = []
        self._last_timestep = 0

    def update(self, stage: float, num_timesteps: int) -> None:
        delta = max(int(num_timesteps) - self._last_timestep, 0)
        self._last_timestep = int(num_timesteps)
        key = round(float(stage), 4)
        self.timesteps_per_stage[key] = self.timesteps_per_stage.get(key, 0) + delta

    def record_transition(self, stage: float, num_timesteps: int) -> None:
        self.stage_transitions.append(
            {"stage": float(stage), "timestep": float(num_timesteps)}
        )

    def summary(self, final_stage: float) -> Dict[str, Any]:
        total = float(sum(self.timesteps_per_stage.values())) or 1.0
        return {
            "final_stage": float(final_stage),
            "reached_final_stage": bool(final_stage >= 1.0 - 1e-6),
            "timesteps_per_stage": dict(sorted(self.timesteps_per_stage.items())),
            "fraction_per_stage": {
                stage: steps / total
                for stage, steps in sorted(self.timesteps_per_stage.items())
            },
            "stage_transitions": list(self.stage_transitions),
        }


class LinearCurriculumCallback(BaseCallback):
    """Increase environment difficulty linearly with timesteps."""

    def __init__(self, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.current_stage = 0.0
        self.tracker = StageTracker()

    def _on_step(self) -> bool:
        progress = min(float(self.num_timesteps) / float(self.total_timesteps), 1.0)
        self.tracker.update(self.current_stage, self.num_timesteps)
        self.current_stage = progress
        _set_stage_on_envs(self.training_env, progress)
        self.logger.record("curriculum/stage", progress)
        return True

    def stage_summary(self) -> Dict[str, Any]:
        return self.tracker.summary(self.current_stage)


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
        self.fractions_at_goal: List[float] = []
        self.visibilities: List[float] = []
        self.collision_events: List[float] = []
        self.progress_rewards: List[float] = []
        self.current_stage = float(self.stages[0]["stage"]) if self.stages else 0.0
        self.tracker = StageTracker()

    def _on_training_start(self) -> None:
        _set_stage_on_envs(self.training_env, self.current_stage)

    def _on_step(self) -> bool:
        infos: Sequence[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.array([]))
        for idx, done in enumerate(dones):
            if not done or idx >= len(infos):
                continue
            info = infos[idx]
            self.successes.append(float(info.get("all_at_goal", False)))
            self.fractions_at_goal.append(float(info.get("fraction_at_goal", 0.0)))
            self.visibilities.append(float(info.get("avg_visibility_ratio", 0.0)))
            self.collision_events.append(float(info.get("collision_event_count", 0.0)))
            self.progress_rewards.append(float(info.get("avg_reward_progress", 0.0)))

        self.tracker.update(self.current_stage, self.num_timesteps)
        next_stage = self._compute_stage()
        if next_stage != self.current_stage:
            self.current_stage = next_stage
            self.tracker.record_transition(next_stage, self.num_timesteps)
            _set_stage_on_envs(self.training_env, self.current_stage)
            if self.verbose:
                print(
                    f"Adaptive curriculum advanced to stage {self.current_stage:.2f} "
                    f"at timestep {self.num_timesteps:,}"
                )
        self.logger.record("curriculum/stage", self.current_stage)
        return True

    def stage_summary(self) -> Dict[str, Any]:
        return self.tracker.summary(self.current_stage)

    def _compute_stage(self) -> float:
        if not self.stages:
            return 0.0

        candidate = float(self.stages[0]["stage"])
        if len(self.successes) < self.warmup_episodes:
            return candidate

        success_rate = rolling_mean(self.successes, self.window)
        fraction_at_goal = rolling_mean(self.fractions_at_goal, self.window)
        visibility_ratio = rolling_mean(self.visibilities, self.window)
        collision_event_count = rolling_mean(self.collision_events, self.window)
        progress_reward = rolling_mean(self.progress_rewards, self.window)

        for stage in self.stages:
            if success_rate < float(stage.get("min_success_rate", 0.0)):
                break
            # Gating on all-or-nothing success stalls the curriculum: under wide
            # randomization that rate stays near zero for a long time, so the
            # agent never reaches the stages where the randomization it needs to
            # generalize is actually switched on. Fraction-delivered moves early
            # and continuously, so it can carry the gate.
            if fraction_at_goal < float(stage.get("min_fraction_at_goal", 0.0)):
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


def _set_stage_on_envs(vec_env: Any, stage: float) -> None:
    """Push the curriculum stage into every worker environment.

    Uses ``env_method`` so this works for ``SubprocVecEnv`` (separate processes)
    and through ``VecNormalize``/``VecMonitor`` wrappers, not just the in-process
    ``DummyVecEnv.envs`` list that the original implementation assumed.
    """
    if vec_env is None:
        return
    try:
        vec_env.env_method("set_curriculum_stage", float(stage))
        return
    except (AttributeError, NotImplementedError):
        pass

    envs = getattr(vec_env, "envs", vec_env)
    for env in envs:
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        if hasattr(inner, "set_curriculum_stage"):
            inner.set_curriculum_stage(stage)


def collect_stage_summary(callbacks: Sequence[Any]) -> Dict[str, Any] | None:
    """Return the curriculum stage summary from whichever callback provides one."""
    for callback in callbacks:
        if hasattr(callback, "stage_summary"):
            return callback.stage_summary()
    return None


class GeneralizationEvalCallback(BaseCallback):
    """Periodically score the policy on a held-out distribution and keep the best.

    Training previously kept whichever weights the last gradient step produced,
    which is a coin flip on a noisy objective. Worse, any tuning done against the
    five named ``unseen_*`` presets leaks them into model selection. This callback
    evaluates on a *procedurally sampled validation split* drawn from the training
    distribution with a disjoint seed stream, leaving the named presets and the
    ``test_procedural`` suite untouched as a genuine test set.

    Selection uses mean fraction-of-flock-delivered rather than the all-or-nothing
    success rate, because the latter is zero for long stretches of training and
    therefore carries no gradient for choosing between checkpoints.
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        model_type: str,
        scenario: str = "validation",
        n_episodes: int = 12,
        eval_freq: int = 25_000,
        seed: int = 10_000,
        best_model_path: Optional[Path] = None,
        training_env: Any = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.env_config = dict(env_config)
        self.model_type = str(model_type)
        self.scenario = str(scenario)
        self.n_episodes = max(int(n_episodes), 1)
        self.eval_freq = max(int(eval_freq), 1)
        self.seed = int(seed)
        self.best_model_path = Path(best_model_path) if best_model_path else None
        self._training_env = training_env
        self.best_score = -np.inf
        self.history: List[Dict[str, float]] = []
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        metrics = self._evaluate()
        self.history.append({"timestep": float(self.num_timesteps), **metrics})

        for key, value in metrics.items():
            self.logger.record(f"validation/{key}", value)

        score = metrics["fraction_at_goal"]
        if score > self.best_score:
            self.best_score = score
            if self.best_model_path is not None:
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
        if self.verbose:
            print(
                f"[validation @ {self.num_timesteps:,}] "
                f"SR={metrics['success_rate']:.3f} | "
                f"FracGoal={metrics['fraction_at_goal']:.3f} | "
                f"MeanDist={metrics['mean_dist_to_goal']:.2f} | "
                f"best={self.best_score:.3f}"
            )
        return True

    def _normalizer(self):
        from stable_baselines3.common.vec_env import VecNormalize

        current = self._training_env if self._training_env is not None else self.training_env
        while current is not None:
            if isinstance(current, VecNormalize) and current.norm_obs:
                return current
            current = getattr(current, "venv", None)
        return None

    def _evaluate(self) -> Dict[str, float]:
        from shepherding.research.models import make_research_env

        normalizer = self._normalizer()
        successes: List[float] = []
        fractions: List[float] = []
        distances: List[float] = []
        lengths: List[float] = []

        for episode in range(self.n_episodes):
            env = make_research_env(
                self.env_config, seed=self.seed + episode, scenario=self.scenario
            )
            obs, info = env.reset(
                seed=self.seed + episode, options={"scenario": self.scenario}
            )
            state = None
            episode_start = np.ones((1,), dtype=bool)
            steps = 0
            while True:
                model_obs = normalizer.normalize_obs(obs) if normalizer else obs
                if self.model_type == "recurrent":
                    action, state = self.model.predict(
                        model_obs,
                        state=state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(model_obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            successes.append(float(info.get("all_at_goal", False)))
            fractions.append(float(info.get("fraction_at_goal", 0.0)))
            distances.append(float(info.get("mean_dist_to_goal", 0.0)))
            lengths.append(float(steps))
            env.close()

        return {
            "success_rate": float(np.mean(successes)),
            "fraction_at_goal": float(np.mean(fractions)),
            "mean_dist_to_goal": float(np.mean(distances)),
            "episode_length": float(np.mean(lengths)),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "episodes": self.n_episodes,
            "eval_freq": self.eval_freq,
            "seed": self.seed,
            "best_fraction_at_goal": (
                float(self.best_score) if np.isfinite(self.best_score) else None
            ),
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "history": self.history,
        }
