"""
HerdingEnvV2 – Enhanced shepherding environment.

Three key upgrades over v1 (HerdingEnv-v0):

1. **Limited dog visibility** – the dog only observes sheep within
   ``visibility_radius`` units.  Invisible sheep are sentinel-padded in the
   observation so the vector shape stays fixed.

2. **Static rectangular obstacles** – configurable rectangles block both the
   dog and sheep.  Sheep apply an avoidance force near obstacle edges.

3. **Always-on sheep dynamics** – cohesion, repulsion, and leader-following
   are active at every step (not just when the dog is nearby).  Only the
   *flee* force is proximity-gated.

Register via:
    gym.make("HerdingEnv-v2")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shepherding.utils.geometry import compute_reward
from shepherding.utils.geometry_v2 import (
    clip_to_free_space,
    obstacle_avoidance_force,
    visible_sheep_mask,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Rect = Tuple[float, float, float, float]  # (x_min, y_min, w, h)

# Sentinel value placed in the observation for each *invisible* sheep
_SENTINEL: float = 999.0


class HerdingEnvV2(gym.Env):
    """Enhanced herding environment with limited visibility and obstacles.

    Observation
    -----------
    Flat vector of shape ``(2 + 2 * n_sheep + 4 * n_obstacles,)``::

        [dog_x, dog_y,
         sheep_0_rel_x, sheep_0_rel_y,   ← _SENTINEL if invisible
         …
         sheep_{N-1}_rel_x, sheep_{N-1}_rel_y,
         obs_0_x, obs_0_y, obs_0_w, obs_0_h,  ← normalised by grid_size
         …]

    Sheep entries are sorted by index (not by visibility) so the network
    can learn stable per-sheep associations.  Invisible sheep receive
    ``(_SENTINEL, _SENTINEL)`` as their relative position.

    Action
    ------
    Continuous ``Box(-1, 1, shape=(2,))`` – dog velocity direction, scaled
    by ``dog_speed``.
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
        sheep_speed: float = 0.3,
        flee_radius: float = 5.0,
        cohesion_factor: float = 0.08,
        repulsion_strength: float = 1.0,
        leader_factor: float = 0.03,
        goal: Tuple[float, float] = (18.0, 18.0),
        max_steps: int = 600,
        success_radius: float = 2.0,
        visibility_radius: float = 8.0,
        obstacles: Optional[List[Rect]] = None,
        obstacle_avoidance_threshold: float = 1.5,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Core parameters
        self.grid_size = float(grid_size)
        self.n_sheep = int(n_sheep)
        self.dog_speed = float(dog_speed)
        self.sheep_speed = float(sheep_speed)
        self.flee_radius = float(flee_radius)
        self.cohesion_factor = float(cohesion_factor)
        self.repulsion_strength = float(repulsion_strength)
        self.leader_factor = float(leader_factor)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)

        # v2-specific parameters
        self.visibility_radius = float(visibility_radius)
        self.obstacles: List[Rect] = obstacles if obstacles is not None else _default_obstacles(grid_size)
        self.obstacle_avoidance_threshold = float(obstacle_avoidance_threshold)
        self.render_mode = render_mode

        # Observation space: dog(2) + sheep(2*N) + obstacles(4*K)
        n_obs = n_obstacles = len(self.obstacles)
        obs_dim: int = 2 + 2 * self.n_sheep + 4 * n_obs
        self.observation_space = spaces.Box(
            low=-self.grid_size,
            high=_SENTINEL + 1.0,  # headroom for sentinel values
            shape=(obs_dim,),
            dtype=np.float32,
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

        # Place sheep in free space (away from obstacles)
        self.sheep_pos = self._sample_free(self.n_sheep)

        # Dog spawns *behind* the flock (opposite side from goal) but close
        # enough to see some sheep immediately (within visibility_radius).
        flock_centroid = np.mean(self.sheep_pos, axis=0)
        goal_dir = self.goal - flock_centroid
        goal_dist = float(np.linalg.norm(goal_dir))
        if goal_dist > 1e-8:
            away = -goal_dir / goal_dist  # unit vector pointing away from goal
        else:
            away = np.array([0.0, -1.0], dtype=np.float32)

        # Spawn at 60–70 % of visibility radius so the dog sees the flock
        spawn_dist = min(self.flee_radius * 0.6, self.visibility_radius * 0.65)
        dog_raw = (flock_centroid + away * spawn_dist).astype(np.float32)
        dog_raw = np.clip(dog_raw, 0.5, self.grid_size - 0.5)
        # Resolve any obstacle collision from the centroid outward
        self.dog_pos = clip_to_free_space(
            np.clip(flock_centroid, 0.5, self.grid_size - 0.5).astype(np.float32),
            dog_raw,
            self.obstacles,
            self.grid_size,
        )
        self.current_step = 0

        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # --- Dog movement -------------------------------------------------
        action = np.asarray(action, dtype=np.float32).flatten()[:2]
        norm = float(np.linalg.norm(action))
        if norm > 1e-8:
            action = action / norm
        proposed_dog = self.dog_pos + action * self.dog_speed
        self.dog_pos = clip_to_free_space(
            self.dog_pos, proposed_dog, self.obstacles, self.grid_size
        )

        # --- Sheep movement -----------------------------------------------
        self._update_sheep()

        # --- Reward -------------------------------------------------------
        reward: float = compute_reward(
            self.sheep_pos, self.dog_pos, self.goal,
            grid_size=self.grid_size,
        )
        # Penalty for each invisible sheep (encourages dog to stay near flock)
        vis_mask = visible_sheep_mask(self.dog_pos, self.sheep_pos, self.visibility_radius)
        n_invisible = int(np.sum(~vis_mask))
        reward -= 0.02 * n_invisible

        # --- Termination --------------------------------------------------
        dists_to_goal = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        all_at_goal = bool(np.all(dists_to_goal < self.success_radius))
        terminated = all_at_goal
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += 100.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Sheep physics (v2 – always-on)
    # ------------------------------------------------------------------
    def _update_sheep(self) -> None:
        """Apply Strömbom-style forces to every sheep.

        Differences from v1:
        * Cohesion and peer-repulsion are **always active** (unconditional).
        * Leader-following: weakly pull all non-leader sheep toward the
          sheep closest to the flock centroid.
        * Obstacle avoidance force applied at every step.
        * Flee only triggers inside ``flee_radius``.
        """
        centroid = np.mean(self.sheep_pos, axis=0)

        # Identify leader (sheep closest to centroid)
        dists_to_centroid = np.linalg.norm(self.sheep_pos - centroid, axis=1)
        leader_idx = int(np.argmin(dists_to_centroid))
        leader_pos = self.sheep_pos[leader_idx]

        for i in range(self.n_sheep):
            velocity = np.zeros(2, dtype=np.float32)

            # 1. Flee from dog if within flee_radius
            diff_dog = self.sheep_pos[i] - self.dog_pos
            dist_dog = float(np.linalg.norm(diff_dog))
            if dist_dog < self.flee_radius and dist_dog > 1e-8:
                velocity += (diff_dog / dist_dog)

            # 2. Cohesion – always active
            diff_centroid = centroid - self.sheep_pos[i]
            velocity += self.cohesion_factor * diff_centroid

            # 3. Peer repulsion – always active
            for j in range(self.n_sheep):
                if j == i:
                    continue
                diff_ij = self.sheep_pos[i] - self.sheep_pos[j]
                dist_ij = float(np.linalg.norm(diff_ij))
                if dist_ij < self.repulsion_strength and dist_ij > 1e-8:
                    velocity += (diff_ij / dist_ij) * (
                        self.repulsion_strength - dist_ij
                    )

            # 4. Leader following (non-leaders only)
            if i != leader_idx:
                diff_leader = leader_pos - self.sheep_pos[i]
                velocity += self.leader_factor * diff_leader

            # 5. Obstacle avoidance force
            velocity += obstacle_avoidance_force(
                self.sheep_pos[i],
                self.obstacles,
                threshold=self.obstacle_avoidance_threshold,
            )

            # 6. Small random drift (always present to keep motion alive)
            velocity += self._rng.normal(0, 0.02, size=2).astype(np.float32)

            # Normalise to sheep_speed
            speed = float(np.linalg.norm(velocity))
            if speed > 1e-8:
                velocity = velocity / speed * self.sheep_speed

            proposed = self.sheep_pos[i] + velocity
            self.sheep_pos[i] = clip_to_free_space(
                self.sheep_pos[i], proposed, self.obstacles, self.grid_size
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_free(self, n: int) -> np.ndarray:
        """Sample *n* positions in free space (not inside any obstacle)."""
        positions = np.empty((n, 2), dtype=np.float32)
        for k in range(n):
            while True:
                p = self._rng.uniform(3.0, self.grid_size - 3.0, size=(2,)).astype(np.float32)
                free = all(
                    not (rx < p[0] < rx + rw and ry < p[1] < ry + rh)
                    for (rx, ry, rw, rh) in self.obstacles
                )
                if free:
                    positions[k] = p
                    break
        return positions

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector.

        Layout: [dog_x, dog_y, sheep_0_rel_x, sheep_0_rel_y, …,
                 obs_0_x_norm, obs_0_y_norm, obs_0_w_norm, obs_0_h_norm, …]
        """
        vis_mask = visible_sheep_mask(
            self.dog_pos, self.sheep_pos, self.visibility_radius
        )

        # Sheep block: relative positions, sentinel for invisible
        sheep_flat = np.full(2 * self.n_sheep, _SENTINEL, dtype=np.float32)
        for i in range(self.n_sheep):
            if vis_mask[i]:
                rel = (self.sheep_pos[i] - self.dog_pos)
                sheep_flat[2 * i] = rel[0]
                sheep_flat[2 * i + 1] = rel[1]

        # Obstacle block: normalised to [0, 1]
        if self.obstacles:
            obs_block = np.array(
                [coord for (rx, ry, rw, rh) in self.obstacles
                 for coord in (rx / self.grid_size, ry / self.grid_size,
                               rw / self.grid_size, rh / self.grid_size)],
                dtype=np.float32,
            )
        else:
            obs_block = np.array([], dtype=np.float32)

        return np.concatenate([self.dog_pos, sheep_flat, obs_block]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        dists = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        vis_mask = visible_sheep_mask(
            self.dog_pos, self.sheep_pos, self.visibility_radius
        )
        return {
            "step": self.current_step,
            "mean_dist_to_goal": float(np.mean(dists)),
            "all_at_goal": bool(np.all(dists < self.success_radius)),
            "n_visible_sheep": int(np.sum(vis_mask)),
        }


# ---------------------------------------------------------------------------
# Default obstacle layout
# ---------------------------------------------------------------------------

def _default_obstacles(grid_size: float) -> List[Rect]:
    """Six small scattered obstacles that mimic natural field features
    (rocks, water troughs, fence posts).  Each is a small rectangle.
    """
    g = grid_size
    return [
        # (x_min, y_min, width, height) – all small relative to grid
        (g * 0.12, g * 0.40, g * 0.06, g * 0.04),   # rock cluster – left
        (g * 0.42, g * 0.20, g * 0.05, g * 0.05),   # rock – lower centre
        (g * 0.62, g * 0.48, g * 0.04, g * 0.06),   # fence post – right mid
        (g * 0.30, g * 0.65, g * 0.06, g * 0.04),   # trough – upper left
        (g * 0.70, g * 0.28, g * 0.04, g * 0.05),   # rock – right lower
        (g * 0.52, g * 0.72, g * 0.05, g * 0.04),   # post – upper right
    ]
