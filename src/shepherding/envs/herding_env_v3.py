"""
Research-oriented shepherding environment with domain randomization.

This environment keeps the core v2 ideas intact while adding:
* partial observability with fixed-size masking
* train-time domain randomization over goals, obstacle layouts, and sheep dynamics
* deterministic unseen evaluation scenario families
* richer episode info for experiment tracking and analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from shepherding.scenarios import (
    default_spawn_bounds,
    fixed_training_obstacles,
    opposite_goal_spawn_bounds,
    scenario_presets,
    structured_training_obstacle_layouts,
)
from shepherding.utils.geometry import compute_convex_hull, compute_reward
from shepherding.utils.geometry_v2 import (
    clip_to_free_space,
    obstacle_avoidance_forces,
    visible_sheep_mask,
)


Rect = Tuple[float, float, float, float]
_SENTINEL: float = 999.0


@dataclass(frozen=True)
class ScenarioConfig:
    goal: Tuple[float, float]
    visibility_radius: float
    obstacles: Tuple[Rect, ...]
    sheep_speed: float
    cohesion_factor: float
    repulsion_strength: float
    leader_factor: float
    obstacle_avoidance_threshold: float
    visibility_radius_range: Optional[Tuple[float, float]] = None
    spawn_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None


class HerdingEnvV3(gym.Env):
    """Research environment for recurrent PPO and generalization studies."""

    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: float = 20.0,
        n_sheep: int = 10,
        dog_speed: float = 1.0,
        sheep_speed: float = 0.32,
        flee_radius: float = 5.5,
        cohesion_factor: float = 0.07,
        repulsion_strength: float = 1.0,
        leader_factor: float = 0.04,
        goal: Tuple[float, float] = (18.0, 18.0),
        max_steps: int = 700,
        success_radius: float = 2.0,
        visibility_radius: float = 7.5,
        max_obstacles: int = 8,
        obstacle_size_range: Tuple[float, float] = (0.8, 1.8),
        obstacle_avoidance_threshold: float = 1.6,
        domain_randomization: bool = True,
        randomize_visibility: bool = True,
        randomize_goal: bool = True,
        randomize_obstacles: bool = True,
        randomize_dynamics: bool = True,
        compute_expensive_metrics: bool = False,
        curriculum_mode: bool = True,
        visibility_loss_penalty: float = 0.05,
        visibility_gain_reward: float = 0.8,
        collision_penalty: float = 0.2,
        w_drive_position: float = 0.35,
        drive_position_scale: float = 0.7,
        zero_visibility_penalty: float = 0.25,
        progress_reward_scale: float = 3.0,
        worst_sheep_reward_scale: float = 2.5,
        persistent_collision_scale: float = 0.15,
        goal_seek_factor: float = 0.03,
        structured_train_obstacles: bool = False,
        opposite_goal_spawn: bool = False,
        scenario: str = "train",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.grid_size = float(grid_size)
        self.n_sheep = int(n_sheep)
        self.dog_speed = float(dog_speed)
        self.base_sheep_speed = float(sheep_speed)
        self.flee_radius = float(flee_radius)
        self.base_cohesion_factor = float(cohesion_factor)
        self.base_repulsion_strength = float(repulsion_strength)
        self.base_leader_factor = float(leader_factor)
        self.base_goal = np.asarray(goal, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.success_radius = float(success_radius)
        self.base_visibility_radius = float(visibility_radius)
        self.max_obstacles = int(max_obstacles)
        self.obstacle_size_range = obstacle_size_range
        self.base_obstacle_avoidance_threshold = float(obstacle_avoidance_threshold)
        self.domain_randomization = bool(domain_randomization)
        self.randomize_visibility = bool(randomize_visibility)
        self.randomize_goal = bool(randomize_goal)
        self.randomize_obstacles = bool(randomize_obstacles)
        self.randomize_dynamics = bool(randomize_dynamics)
        self.compute_expensive_metrics = bool(compute_expensive_metrics)
        self.curriculum_mode = bool(curriculum_mode)
        self.visibility_loss_penalty = float(visibility_loss_penalty)
        self.visibility_gain_reward = float(visibility_gain_reward)
        self.collision_penalty = float(collision_penalty)
        self.w_drive_position = float(w_drive_position)
        self.drive_position_scale = float(drive_position_scale)
        self.zero_visibility_penalty = float(zero_visibility_penalty)
        self.progress_reward_scale = float(progress_reward_scale)
        self.worst_sheep_reward_scale = float(worst_sheep_reward_scale)
        self.persistent_collision_scale = float(persistent_collision_scale)
        self.goal_seek_factor = float(goal_seek_factor)
        self.structured_train_obstacles = bool(structured_train_obstacles)
        self.opposite_goal_spawn = bool(opposite_goal_spawn)
        self.scenario = scenario
        self.render_mode = render_mode

        self.goal = self.base_goal.copy()
        self.visibility_radius = self.base_visibility_radius
        self.sheep_speed = self.base_sheep_speed
        self.cohesion_factor = self.base_cohesion_factor
        self.repulsion_strength = self.base_repulsion_strength
        self.leader_factor = self.base_leader_factor
        self.obstacle_avoidance_threshold = self.base_obstacle_avoidance_threshold
        self.obstacles: List[Rect] = []
        self.spawn_bounds = default_spawn_bounds()

        obs_dim = 4 + 2 * self.n_sheep + 4 * self.max_obstacles
        self.observation_space = spaces.Box(
            low=-self.grid_size,
            high=_SENTINEL + 1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.dog_pos = np.zeros(2, dtype=np.float32)
        self.sheep_pos = np.zeros((self.n_sheep, 2), dtype=np.float32)
        self.current_step = 0
        self._rng: np.random.Generator = np.random.default_rng()
        self._episode_returns = 0.0
        self._visibility_sum = 0.0
        self._dog_path_length = 0.0
        self._collision_count = 0
        self._collision_event_count = 0
        self._prev_collided = False
        self._last_scenario_name = scenario
        self._prev_visible_ratio = 1.0
        self._prev_mean_dist_to_goal = 0.0
        self._prev_max_dist_to_goal = 0.0
        self._last_progress_delta = 0.0
        self._reward_component_sums: Dict[str, float] = {
            "base": 0.0,
            "progress": 0.0,
            "worst_sheep": 0.0,
            "visibility_loss": 0.0,
            "visibility_gain": 0.0,
            "zero_visibility": 0.0,
            "stray": 0.0,
            "drive": 0.0,
            "collision": 0.0,
            "success_bonus": 0.0,
        }
        self._last_reward_terms: Dict[str, float] = {
            "base": 0.0,
            "progress": 0.0,
            "worst_sheep": 0.0,
            "visibility_loss": 0.0,
            "visibility_gain": 0.0,
            "zero_visibility": 0.0,
            "stray": 0.0,
            "drive": 0.0,
            "collision": 0.0,
            "success_bonus": 0.0,
        }
        self.curriculum_stage = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        scenario_name = self._resolve_scenario_name(options)
        config = self._sample_scenario(scenario_name)
        self._apply_config(config)
        self._last_scenario_name = scenario_name

        self.sheep_pos = self._sample_flock_positions()
        self.dog_pos = self._spawn_dog()

        self.current_step = 0
        self._episode_returns = 0.0
        self._visibility_sum = 0.0
        self._dog_path_length = 0.0
        self._collision_count = 0
        self._collision_event_count = 0
        self._prev_collided = False
        initial_vis_mask = visible_sheep_mask(
            self.dog_pos, self.sheep_pos, self.visibility_radius
        )
        self._prev_visible_ratio = float(np.mean(initial_vis_mask))
        initial_dists_to_goal = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        self._prev_mean_dist_to_goal = float(np.mean(initial_dists_to_goal))
        self._prev_max_dist_to_goal = float(np.max(initial_dists_to_goal))
        self._last_progress_delta = 0.0
        for key in self._reward_component_sums:
            self._reward_component_sums[key] = 0.0
            self._last_reward_terms[key] = 0.0

        obs = self._get_obs(initial_vis_mask)
        info = self._get_info(
            vis_mask=initial_vis_mask,
            dists_to_goal=initial_dists_to_goal,
        )
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32).flatten()[:2]
        norm = float(np.linalg.norm(action))
        if norm > 1e-8:
            action = action / norm

        prev_dog_pos = self.dog_pos.copy()
        proposed_dog = prev_dog_pos + action * self.dog_speed
        clipped_dog = clip_to_free_space(
            prev_dog_pos, proposed_dog, self.obstacles, self.grid_size
        )
        collided = np.linalg.norm(clipped_dog - proposed_dog) > 1e-6
        if collided:
            self._collision_count += 1
            if not self._prev_collided:
                self._collision_event_count += 1
        self.dog_pos = clipped_dog
        self._dog_path_length += float(np.linalg.norm(self.dog_pos - prev_dog_pos))

        self._update_sheep()

        centroid = np.mean(self.sheep_pos, axis=0)
        dist_to_centroid = np.linalg.norm(self.sheep_pos - centroid, axis=1)

        base_reward = compute_reward(
            self.sheep_pos,
            self.dog_pos,
            self.goal,
            grid_size=self.grid_size,
        )
        dists_to_goal = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        mean_dist_to_goal = float(np.mean(dists_to_goal))
        max_dist_to_goal = float(np.max(dists_to_goal))
        progress_delta = self._prev_mean_dist_to_goal - mean_dist_to_goal
        progress_reward = self.progress_reward_scale * progress_delta
        worst_sheep_delta = self._prev_max_dist_to_goal - max_dist_to_goal
        worst_sheep_reward = self.worst_sheep_reward_scale * worst_sheep_delta

        vis_mask = visible_sheep_mask(self.dog_pos, self.sheep_pos, self.visibility_radius)
        visible_ratio = float(np.mean(vis_mask))
        self._visibility_sum += visible_ratio
        visibility_loss_term = -self.visibility_loss_penalty * float(1.0 - visible_ratio)
        visibility_gain_term = self.visibility_gain_reward * (
            visible_ratio - self._prev_visible_ratio
        )
        zero_visibility_term = 0.0
        if np.sum(vis_mask) == 0:
            zero_visibility_term = -self.zero_visibility_penalty

        strays = int(np.sum(dist_to_centroid > self.success_radius * 1.8))
        stray_penalty = -0.05 * strays
        drive_term = self.w_drive_position * self._drive_position_reward(centroid)
        collision_term = 0.0
        if collided:
            if self._prev_collided:
                collision_term = -self.collision_penalty * self.persistent_collision_scale
            else:
                collision_term = -self.collision_penalty

        reward = (
            base_reward
            + progress_reward
            + worst_sheep_reward
            + visibility_loss_term
            + visibility_gain_term
            + zero_visibility_term
            + stray_penalty
            + drive_term
            + collision_term
        )
        self._last_reward_terms = {
            "base": float(base_reward),
            "progress": float(progress_reward),
            "worst_sheep": float(worst_sheep_reward),
            "visibility_loss": float(visibility_loss_term),
            "visibility_gain": float(visibility_gain_term),
            "zero_visibility": float(zero_visibility_term),
            "stray": float(stray_penalty),
            "drive": float(drive_term),
            "collision": float(collision_term),
            "success_bonus": 0.0,
        }
        for key, value in self._last_reward_terms.items():
            self._reward_component_sums[key] += value

        self._prev_visible_ratio = visible_ratio
        self._prev_mean_dist_to_goal = mean_dist_to_goal
        self._prev_max_dist_to_goal = max_dist_to_goal
        self._last_progress_delta = float(progress_delta)
        self._prev_collided = bool(collided)
        self._episode_returns += reward

        all_at_goal = bool(np.all(dists_to_goal < self.success_radius))
        terminated = all_at_goal
        truncated = self.current_step >= self.max_steps

        if terminated:
            success_bonus = 125.0
            reward += success_bonus
            self._episode_returns += success_bonus
            self._last_reward_terms["success_bonus"] = float(success_bonus)
            self._reward_component_sums["success_bonus"] += float(success_bonus)

        return (
            self._get_obs(vis_mask),
            reward,
            terminated,
            truncated,
            self._get_info(
                dists_to_goal=dists_to_goal,
                vis_mask=vis_mask,
                centroid=centroid,
                stray_count=strays,
                collided=collided,
                is_terminal=terminated or truncated,
            ),
        )

    def _resolve_scenario_name(self, options: Optional[Dict[str, Any]]) -> str:
        if options and "scenario" in options:
            return str(options["scenario"])
        return self.scenario

    def _sample_scenario(self, scenario_name: str) -> ScenarioConfig:
        if scenario_name == "train":
            goal_randomization = self.randomize_goal
            visibility_randomization = self.randomize_visibility
            obstacle_randomization = self.randomize_obstacles
            dynamics_randomization = self.randomize_dynamics
            visibility_range = (5.5, 9.0)

            if self.curriculum_mode:
                if self.curriculum_stage < 0.33:
                    goal_randomization = False
                    obstacle_randomization = False
                    dynamics_randomization = False
                    visibility_randomization = True
                    visibility_range = (
                        max(5.5, self.base_visibility_radius - 0.5),
                        self.base_visibility_radius + 0.5,
                    )
                elif self.curriculum_stage < 0.66:
                    goal_randomization = True
                    obstacle_randomization = False
                    dynamics_randomization = False
                    visibility_randomization = True
                    visibility_range = (5.8, 8.2)

            if not self.domain_randomization:
                goal = tuple(float(v) for v in self.base_goal)
                obstacles = tuple(fixed_training_obstacles(self.grid_size))
                if self.structured_train_obstacles:
                    layouts = structured_training_obstacle_layouts(self.grid_size)
                    obstacles = tuple(layouts[int(self._rng.integers(0, len(layouts)))])
                return ScenarioConfig(
                    goal=goal,
                    visibility_radius=self.base_visibility_radius,
                    obstacles=obstacles,
                    sheep_speed=self.base_sheep_speed,
                    cohesion_factor=self.base_cohesion_factor,
                    repulsion_strength=self.base_repulsion_strength,
                    leader_factor=self.base_leader_factor,
                    obstacle_avoidance_threshold=self.base_obstacle_avoidance_threshold,
                    visibility_radius_range=None,
                    spawn_bounds=(
                        opposite_goal_spawn_bounds(goal, self.grid_size)
                        if self.opposite_goal_spawn
                        else default_spawn_bounds()
                    ),
                )
            goal = tuple(float(v) for v in self.base_goal)
            if goal_randomization:
                goal = tuple(
                    self._rng.uniform(
                        self.grid_size * 0.65, self.grid_size * 0.92, size=2
                    ).tolist()
                )

            visibility_radius = self.base_visibility_radius
            if visibility_randomization:
                visibility_radius = float(self._rng.uniform(*visibility_range))

            obstacles: Tuple[Rect, ...]
            if obstacle_randomization:
                if self.structured_train_obstacles:
                    layouts = structured_training_obstacle_layouts(self.grid_size)
                    obstacles = tuple(layouts[int(self._rng.integers(0, len(layouts)))])
                else:
                    obstacle_count = int(self._rng.integers(3, self.max_obstacles + 1))
                    obstacles = tuple(
                        self._generate_obstacles(count=obstacle_count, center_bias=False)
                    )
            else:
                if self.structured_train_obstacles:
                    layouts = structured_training_obstacle_layouts(self.grid_size)
                    obstacles = tuple(layouts[int(self._rng.integers(0, len(layouts)))])
                else:
                    obstacles = tuple(fixed_training_obstacles(self.grid_size))

            sheep_speed = self.base_sheep_speed
            cohesion_factor = self.base_cohesion_factor
            repulsion_strength = self.base_repulsion_strength
            leader_factor = self.base_leader_factor
            obstacle_threshold = self.base_obstacle_avoidance_threshold
            if dynamics_randomization:
                sheep_speed = float(self._rng.uniform(0.28, 0.38))
                cohesion_factor = float(self._rng.uniform(0.05, 0.10))
                repulsion_strength = float(self._rng.uniform(0.9, 1.4))
                leader_factor = float(self._rng.uniform(0.02, 0.06))
                obstacle_threshold = float(self._rng.uniform(1.2, 2.1))

            return ScenarioConfig(
                goal=goal,
                visibility_radius=visibility_radius,
                obstacles=obstacles,
                sheep_speed=sheep_speed,
                cohesion_factor=cohesion_factor,
                repulsion_strength=repulsion_strength,
                leader_factor=leader_factor,
                obstacle_avoidance_threshold=obstacle_threshold,
                visibility_radius_range=visibility_range if visibility_randomization else None,
                spawn_bounds=(
                    opposite_goal_spawn_bounds(goal, self.grid_size)
                    if self.opposite_goal_spawn
                    else default_spawn_bounds()
                ),
            )

        presets = scenario_presets(self.grid_size)
        if scenario_name not in presets:
            raise ValueError(f"Unknown scenario '{scenario_name}'.")
        preset = presets[scenario_name]
        return ScenarioConfig(
            goal=preset.goal,
            visibility_radius=preset.visibility_radius,
            obstacles=preset.obstacles,
            sheep_speed=preset.sheep_speed,
            cohesion_factor=preset.cohesion_factor,
            repulsion_strength=preset.repulsion_strength,
            leader_factor=preset.leader_factor,
            obstacle_avoidance_threshold=preset.obstacle_avoidance_threshold,
            spawn_bounds=preset.spawn_bounds or default_spawn_bounds(),
        )

    def _apply_config(self, config: ScenarioConfig) -> None:
        self.goal = np.asarray(config.goal, dtype=np.float32)
        self.visibility_radius = float(config.visibility_radius)
        self.obstacles = list(config.obstacles)
        self.sheep_speed = float(config.sheep_speed)
        self.cohesion_factor = float(config.cohesion_factor)
        self.repulsion_strength = float(config.repulsion_strength)
        self.leader_factor = float(config.leader_factor)
        self.obstacle_avoidance_threshold = float(config.obstacle_avoidance_threshold)
        self.spawn_bounds = config.spawn_bounds or default_spawn_bounds()

    def _sample_flock_positions(self) -> np.ndarray:
        (x_low, y_low), (x_high, y_high) = self.spawn_bounds
        centroid = self._rng.uniform(
            low=np.array([self.grid_size * x_low, self.grid_size * y_low]),
            high=np.array([self.grid_size * x_high, self.grid_size * y_high]),
            size=(2,),
        ).astype(np.float32)
        positions = np.empty((self.n_sheep, 2), dtype=np.float32)
        for idx in range(self.n_sheep):
            for _ in range(256):
                candidate = centroid + self._rng.normal(0.0, 1.25, size=2).astype(np.float32)
                candidate = np.clip(candidate, 1.0, self.grid_size - 1.0)
                if self._is_free(candidate):
                    positions[idx] = candidate
                    break
            else:
                positions[idx] = centroid
        return positions

    def _spawn_dog(self) -> np.ndarray:
        flock_centroid = np.mean(self.sheep_pos, axis=0)
        goal_dir = self.goal - flock_centroid
        goal_dist = float(np.linalg.norm(goal_dir))
        if goal_dist > 1e-8:
            away = -goal_dir / goal_dist
        else:
            away = np.array([-1.0, -1.0], dtype=np.float32) / np.sqrt(2.0)
        spawn_dist = min(self.visibility_radius * 0.7, self.flee_radius * 0.9)
        candidate = flock_centroid + away * spawn_dist
        candidate = np.clip(candidate, 0.8, self.grid_size - 0.8).astype(np.float32)
        return clip_to_free_space(
            np.clip(flock_centroid, 0.8, self.grid_size - 0.8).astype(np.float32),
            candidate,
            self.obstacles,
            self.grid_size,
        )

    def _update_sheep(self) -> None:
        centroid = np.mean(self.sheep_pos, axis=0)
        dists_to_centroid = np.linalg.norm(self.sheep_pos - centroid, axis=1)
        leader_idx = int(np.argmin(dists_to_centroid))
        leader_pos = self.sheep_pos[leader_idx]
        diff_dog = self.sheep_pos - self.dog_pos
        dog_dists = np.linalg.norm(diff_dog, axis=1)
        flee = np.zeros_like(self.sheep_pos, dtype=np.float32)
        flee_mask = (dog_dists < self.flee_radius) & (dog_dists > 1e-8)
        flee[flee_mask] = diff_dog[flee_mask] / dog_dists[flee_mask, None]

        cohesion = self.cohesion_factor * (centroid - self.sheep_pos)

        pairwise_diff = self.sheep_pos[:, None, :] - self.sheep_pos[None, :, :]
        pairwise_dist_sq = np.sum(pairwise_diff * pairwise_diff, axis=2)
        repulsion_mask = (
            (pairwise_dist_sq < self.repulsion_strength * self.repulsion_strength)
            & (pairwise_dist_sq > 1e-12)
        )
        pairwise_dist = np.zeros_like(pairwise_dist_sq, dtype=np.float32)
        pairwise_dist[repulsion_mask] = np.sqrt(pairwise_dist_sq[repulsion_mask]).astype(
            np.float32
        )
        repulsion_weights = np.zeros_like(pairwise_dist_sq, dtype=np.float32)
        repulsion_weights[repulsion_mask] = (
            (self.repulsion_strength - pairwise_dist[repulsion_mask])
            / pairwise_dist[repulsion_mask]
        )
        repulsion = np.sum(
            pairwise_diff * repulsion_weights[..., None],
            axis=1,
            dtype=np.float32,
        )

        leader = self.leader_factor * (leader_pos - self.sheep_pos)
        leader[leader_idx] = 0.0

        if len(self.obstacles) > 0:
            obstacle_force = obstacle_avoidance_forces(
                self.sheep_pos,
                self.obstacles,
                threshold=self.obstacle_avoidance_threshold,
            )
        else:
            obstacle_force = np.zeros_like(self.sheep_pos, dtype=np.float32)
        noise = self._rng.normal(0.0, 0.02, size=self.sheep_pos.shape).astype(np.float32)

        velocities = flee + cohesion + repulsion + leader + obstacle_force + noise
        goal_diff = self.goal - self.sheep_pos
        goal_dists = np.linalg.norm(goal_diff, axis=1)
        pressured_mask = (dog_dists < self.flee_radius) & (goal_dists > 1e-8)
        if np.any(pressured_mask):
            velocities[pressured_mask] += (
                self.goal_seek_factor
                * goal_diff[pressured_mask]
                / goal_dists[pressured_mask, None]
            )
        speeds = np.linalg.norm(velocities, axis=1)
        moving_mask = speeds > 1e-8
        velocities[moving_mask] = (
            velocities[moving_mask] / speeds[moving_mask, None] * self.sheep_speed
        )

        for i in range(self.n_sheep):
            proposed = self.sheep_pos[i] + velocities[i]
            self.sheep_pos[i] = clip_to_free_space(
                self.sheep_pos[i], proposed, self.obstacles, self.grid_size
            )

    def _distances_to_centroid(self) -> np.ndarray:
        centroid = np.mean(self.sheep_pos, axis=0)
        return np.linalg.norm(self.sheep_pos - centroid, axis=1)

    def _drive_position_reward(self, centroid: np.ndarray) -> float:
        goal_vec = self.goal - centroid
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= 1e-8:
            return 0.0
        drive_distance = min(
            self.visibility_radius * self.drive_position_scale,
            self.flee_radius * 0.9,
        )
        drive_target = centroid - (goal_vec / goal_dist) * drive_distance
        max_dist = float(np.sqrt(2.0) * self.grid_size)
        return 1.0 - (float(np.linalg.norm(self.dog_pos - drive_target)) / max_dist)

    def _get_obs(self, vis_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if vis_mask is None:
            vis_mask = visible_sheep_mask(
                self.dog_pos, self.sheep_pos, self.visibility_radius
            )
        sheep_flat = np.full(2 * self.n_sheep, _SENTINEL, dtype=np.float32)
        if np.any(vis_mask):
            visible_idx = np.flatnonzero(vis_mask)
            rel = self.sheep_pos[visible_idx] - self.dog_pos
            sheep_flat[2 * visible_idx] = rel[:, 0]
            sheep_flat[2 * visible_idx + 1] = rel[:, 1]

        goal_rel = self.goal - self.dog_pos
        obstacle_block = np.full(4 * self.max_obstacles, -1.0, dtype=np.float32)
        for idx, (rx, ry, rw, rh) in enumerate(self.obstacles[: self.max_obstacles]):
            base = 4 * idx
            obstacle_block[base : base + 4] = np.array(
                [
                    rx / self.grid_size,
                    ry / self.grid_size,
                    rw / self.grid_size,
                    rh / self.grid_size,
                ],
                dtype=np.float32,
            )
        return np.concatenate([self.dog_pos, goal_rel, sheep_flat, obstacle_block]).astype(
            np.float32
        )

    def _get_info(
        self,
        *,
        dists_to_goal: Optional[np.ndarray] = None,
        vis_mask: Optional[np.ndarray] = None,
        centroid: Optional[np.ndarray] = None,
        stray_count: Optional[int] = None,
        collided: bool = False,
        is_terminal: bool = False,
    ) -> Dict[str, Any]:
        if dists_to_goal is None:
            dists_to_goal = np.linalg.norm(self.sheep_pos - self.goal, axis=1)
        if vis_mask is None:
            vis_mask = visible_sheep_mask(
                self.dog_pos, self.sheep_pos, self.visibility_radius
            )
        if centroid is None:
            centroid = np.mean(self.sheep_pos, axis=0)
        if stray_count is None:
            stray_count = int(
                np.sum(self._distances_to_centroid() > self.success_radius * 1.8)
            )

        hull_area = 0.0
        if self.compute_expensive_metrics or is_terminal:
            hull = compute_convex_hull(self.sheep_pos)
            hull_area = float(hull.volume) if hull is not None else 0.0

        visible_ratio = float(np.mean(vis_mask))
        step_count = max(self.current_step, 1)
        info = {
            "step": self.current_step,
            "scenario": self._last_scenario_name,
            "mean_dist_to_goal": float(np.mean(dists_to_goal)),
            "all_at_goal": bool(np.all(dists_to_goal < self.success_radius)),
            "n_visible_sheep": int(np.sum(vis_mask)),
            "visible_ratio": visible_ratio,
            "flock_hull_area": hull_area,
            "flock_centroid_x": float(centroid[0]),
            "flock_centroid_y": float(centroid[1]),
            "stray_count": stray_count,
            "collision_count": self._collision_count,
            "collision_event_count": self._collision_event_count,
            "collided": bool(collided),
            "dog_path_length": self._dog_path_length,
            "episode_return": self._episode_returns,
            "avg_visibility_ratio": (
                self._visibility_sum / step_count
            ),
            "progress_to_goal": self._last_progress_delta,
            "reward_base": self._last_reward_terms["base"],
            "reward_progress": self._last_reward_terms["progress"],
            "reward_worst_sheep": self._last_reward_terms["worst_sheep"],
            "reward_visibility_loss": self._last_reward_terms["visibility_loss"],
            "reward_visibility_gain": self._last_reward_terms["visibility_gain"],
            "reward_zero_visibility": self._last_reward_terms["zero_visibility"],
            "reward_stray": self._last_reward_terms["stray"],
            "reward_drive": self._last_reward_terms["drive"],
            "reward_collision": self._last_reward_terms["collision"],
            "reward_success_bonus": self._last_reward_terms["success_bonus"],
            "avg_reward_base": self._reward_component_sums["base"] / step_count,
            "avg_reward_progress": self._reward_component_sums["progress"] / step_count,
            "avg_reward_worst_sheep": self._reward_component_sums["worst_sheep"] / step_count,
            "avg_reward_visibility_loss": (
                self._reward_component_sums["visibility_loss"] / step_count
            ),
            "avg_reward_visibility_gain": (
                self._reward_component_sums["visibility_gain"] / step_count
            ),
            "avg_reward_zero_visibility": (
                self._reward_component_sums["zero_visibility"] / step_count
            ),
            "avg_reward_stray": self._reward_component_sums["stray"] / step_count,
            "avg_reward_drive": self._reward_component_sums["drive"] / step_count,
            "avg_reward_collision": self._reward_component_sums["collision"] / step_count,
            "avg_reward_success_bonus": (
                self._reward_component_sums["success_bonus"] / step_count
            ),
            "curriculum_stage": float(self.curriculum_stage),
        }
        if is_terminal or self.current_step >= self.max_steps or info["all_at_goal"]:
            info["episode"] = {
                "r": self._episode_returns,
                "l": self.current_step,
            }
        return info

    def _generate_obstacles(self, count: int, center_bias: bool) -> List[Rect]:
        obstacles: List[Rect] = []
        for _ in range(count):
            for _ in range(256):
                width = float(self._rng.uniform(*self.obstacle_size_range))
                height = float(self._rng.uniform(*self.obstacle_size_range))
                if center_bias:
                    center = self._rng.uniform(
                        self.grid_size * 0.35, self.grid_size * 0.75, size=2
                    )
                else:
                    center = self._rng.uniform(
                        self.grid_size * 0.15, self.grid_size * 0.85, size=2
                    )
                rx = float(np.clip(center[0] - width / 2.0, 0.6, self.grid_size - width - 0.6))
                ry = float(np.clip(center[1] - height / 2.0, 0.6, self.grid_size - height - 0.6))
                rect = (rx, ry, width, height)
                if self._valid_obstacle(rect, obstacles):
                    obstacles.append(rect)
                    break
        return obstacles

    def _valid_obstacle(self, rect: Rect, existing: Sequence[Rect]) -> bool:
        rx, ry, rw, rh = rect
        for ex, ey, ew, eh in existing:
            no_overlap = rx + rw + 0.5 < ex or ex + ew + 0.5 < rx or ry + rh + 0.5 < ey or ey + eh + 0.5 < ry
            if not no_overlap:
                return False
        start_zone = rx < self.grid_size * 0.48 and ry < self.grid_size * 0.48
        goal_zone = (rx + rw) > self.grid_size * 0.62 and (ry + rh) > self.grid_size * 0.62
        return not (start_zone or goal_zone)

    def _is_free(self, point: np.ndarray) -> bool:
        x, y = float(point[0]), float(point[1])
        if x <= 0.0 or y <= 0.0 or x >= self.grid_size or y >= self.grid_size:
            return False
        for rx, ry, rw, rh in self.obstacles:
            if rx < x < rx + rw and ry < y < ry + rh:
                return False
        return True

    def set_curriculum_stage(self, stage: float) -> None:
        self.curriculum_stage = float(np.clip(stage, 0.0, 1.0))
