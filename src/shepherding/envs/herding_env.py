"""
HerdingEnv – A Gymnasium environment for the shepherding problem.

Implements the Strömbom et al. (2014) sheep–dog interaction model on a
continuous 2-D grid.  One RL-controlled dog must herd *N* heuristic sheep
toward a goal position.

Key configurable parameters (passed via ``env_config``):
    grid_size, n_sheep, dog_speed, sheep_speed, flee_radius,
    cohesion_factor, repulsion_strength, goal, max_steps, success_radius.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shepherding.utils.geometry import compute_reward


class HerdingEnv(gym.Env):
    """Shepherding environment with Strömbom-style sheep physics.

    Observation
    -----------
    Flat vector of shape ``(2 + 2 * n_sheep,)`` containing:
    ``[dog_x, dog_y, sheep_1_x, sheep_1_y, …, sheep_N_x, sheep_N_y]``.

    Action
    ------
    Continuous ``Box(-1, 1, shape=(2,))`` interpreted as the dog's velocity
    direction, scaled by ``dog_speed``.
    """

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        grid_size: float = 20.0,
        n_sheep: int = 10,
        dog_speed: float = 1.0,
        sheep_speed: float = 0.4,
        flee_radius: float = 6.0,
        cohesion_factor: float = 0.05,
        repulsion_strength: float = 1.0,
        goal: Tuple[float, float] = (18.0, 18.0),
        max_steps: int = 500,
        success_radius: float = 2.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Environment parameters
        self.grid_size: float = grid_size
        self.n_sheep: int = n_sheep
        self.dog_speed: float = dog_speed
        self.sheep_speed: float = sheep_speed
        self.flee_radius: float = flee_radius
        self.cohesion_factor: float = cohesion_factor
        self.repulsion_strength: float = repulsion_strength
        self.goal: np.ndarray = np.asarray(goal, dtype=np.float32)
        self.max_steps: int = max_steps
        self.success_radius: float = success_radius
        self.render_mode = render_mode

        # Spaces
        obs_dim: int = 2 + 2 * self.n_sheep
        self.observation_space = spaces.Box(
            low=-self.grid_size, high=self.grid_size, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state (populated by reset)
        self.dog_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.sheep_pos: np.ndarray = np.zeros((self.n_sheep, 2), dtype=np.float32)
        self.current_step: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Sheep scattered in upper half; dog spawns nearby (within 5 units)
        self.sheep_pos = self._rng.uniform(
            5.0, self.grid_size - 5.0, size=(self.n_sheep, 2)
        ).astype(np.float32)
        flock_centroid = np.mean(self.sheep_pos, axis=0)
        dog_offset = self._rng.uniform(-5.0, 5.0, size=(2,)).astype(np.float32)
        self.dog_pos = np.clip(
            flock_centroid + dog_offset, 0.0, self.grid_size
        ).astype(np.float32)
        self.current_step = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # --- Dog movement -------------------------------------------------
        action = np.asarray(action, dtype=np.float32).flatten()[:2]
        norm = np.linalg.norm(action)
        if norm > 1e-8:
            action = action / norm
        self.dog_pos = self.dog_pos + action * self.dog_speed
        self.dog_pos = np.clip(self.dog_pos, 0.0, self.grid_size).astype(np.float32)

        # --- Sheep movement (Strömbom et al. 2014) ------------------------
        self._update_sheep()

        # --- Reward -------------------------------------------------------
        reward: float = compute_reward(
            self.sheep_pos, self.dog_pos, self.goal,
            grid_size=self.grid_size,
        )

        # --- Termination conditions ---------------------------------------
        dists_to_goal = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        all_at_goal: bool = bool(np.all(dists_to_goal < self.success_radius))
        terminated: bool = all_at_goal
        truncated: bool = self.current_step >= self.max_steps

        # Bonus for success
        if terminated:
            reward += 100.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Sheep physics
    # ------------------------------------------------------------------
    def _update_sheep(self) -> None:
        """Apply Strömbom-style forces to every sheep and clamp to grid.

        Forces (cohesion, repulsion, flee) only activate when the dog is
        within ``awareness_radius = 2 * flee_radius``.  This prevents
        unrealistic autonomous drift when the dog is far away.
        """
        centroid = np.mean(self.sheep_pos, axis=0)
        awareness_radius: float = 2.0 * self.flee_radius

        for i in range(self.n_sheep):
            velocity = np.zeros(2, dtype=np.float32)

            diff_dog = self.sheep_pos[i] - self.dog_pos
            dist_dog = float(np.linalg.norm(diff_dog))

            # Only apply behavioural forces when the dog is close enough
            # for the sheep to be "aware" of it.
            if dist_dog < awareness_radius:
                # 1. Flee from dog if within flee_radius
                if dist_dog < self.flee_radius and dist_dog > 1e-8:
                    velocity += (diff_dog / dist_dog)  # unit flee direction

                # 2. Cohesion – attraction toward flock centroid
                diff_centroid = centroid - self.sheep_pos[i]
                velocity += self.cohesion_factor * diff_centroid

                # 3. Repulsion – personal-space avoidance from other sheep
                for j in range(self.n_sheep):
                    if j == i:
                        continue
                    diff_ij = self.sheep_pos[i] - self.sheep_pos[j]
                    dist_ij = float(np.linalg.norm(diff_ij))
                    if dist_ij < self.repulsion_strength and dist_ij > 1e-8:
                        velocity += (diff_ij / dist_ij) * (
                            self.repulsion_strength - dist_ij
                        )

            # Normalise to sheep_speed
            speed = float(np.linalg.norm(velocity))
            if speed > 1e-8:
                velocity = velocity / speed * self.sheep_speed
                # Add small noise only when the sheep is actually moving
                velocity += self._rng.normal(0, 0.005, size=2).astype(np.float32)

            self.sheep_pos[i] = np.clip(
                self.sheep_pos[i] + velocity, 0.0, self.grid_size
            ).astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Observation: dog absolute pos + sheep positions relative to dog."""
        relative_sheep = (self.sheep_pos - self.dog_pos).flatten()
        return np.concatenate(
            [self.dog_pos, relative_sheep]
        ).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        dists = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        return {
            "step": self.current_step,
            "mean_dist_to_goal": float(np.mean(dists)),
            "all_at_goal": bool(np.all(dists < self.success_radius)),
        }
