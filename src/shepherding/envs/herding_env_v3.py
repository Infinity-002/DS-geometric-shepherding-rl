"""
Research-oriented shepherding environment with domain randomization.

This environment keeps the core v2 ideas intact while adding:
* partial observability with fixed-size masking
* train-time domain randomization over goals, obstacle layouts, and sheep dynamics
* deterministic unseen evaluation scenario families
* richer episode info for experiment tracking and analysis
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from shepherding.scenarios.generators import (
    TEST_DYNAMICS_SPEC,
    TEST_LAYOUT_SPEC,
    TRAIN_DYNAMICS_SPEC,
    TRAIN_LAYOUT_SPEC,
    DynamicsSpec,
    sample_goal_and_spawn,
    sample_layout,
)
from shepherding.utils.geometry import compute_convex_hull, compute_reward
from shepherding.utils.geometry_v2 import (
    clip_to_free_space,
    obstacle_avoidance_forces,
    ray_angles,
    raycast_distances,
    visible_sheep_mask,
)


Rect = Tuple[float, float, float, float]

#: Legacy observations mark unseen sheep with this out-of-range value. It is kept
#: only for the heuristic/behavioural-cloning agents that decode that layout; the
#: egocentric layout uses an explicit visibility flag instead so that no input
#: feature ever leaves the normalized range.
_SENTINEL: float = 999.0

#: Number of non-per-sheep, non-lidar features in the egocentric observation.
#: goal(4) + frame-axis boundary clearances(4) + previous action(2)
#: + visible-flock summary(8) + flock memory(4) + centroid-to-goal geometry(3)
#: + time remaining(1)
_EGO_SCALAR_BLOCK: int = 26
_EGO_OBS_BOUND: float = 10.0
#: Horizon (in steps) over which the "time since the flock was last seen" feature
#: saturates.
_MEMORY_HORIZON: float = 64.0


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
    n_sheep: Optional[int] = None
    flee_radius: Optional[float] = None
    observation_noise: float = 0.0
    action_noise: float = 0.0
    topology: str = "preset"


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
        randomize_sheep_count: bool = False,
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
        success_bonus: float = 20.0,
        structured_train_obstacles: bool = False,
        opposite_goal_spawn: bool = False,
        observation_mode: str = "egocentric",
        observation_frame: str = "world",
        n_lidar_rays: int = 24,
        lidar_range: Optional[float] = None,
        max_sheep: Optional[int] = None,
        scenario: str = "train",
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.grid_size = float(grid_size)
        self.base_n_sheep = int(n_sheep)
        self.n_sheep = int(n_sheep)
        self.dog_speed = float(dog_speed)
        self.base_sheep_speed = float(sheep_speed)
        self.base_flee_radius = float(flee_radius)
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
        self.randomize_sheep_count = bool(randomize_sheep_count)
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
        self.success_bonus = float(success_bonus)
        self.structured_train_obstacles = bool(structured_train_obstacles)
        self.opposite_goal_spawn = bool(opposite_goal_spawn)
        self.scenario = scenario
        self.render_mode = render_mode

        self.observation_mode = str(observation_mode).lower()
        if self.observation_mode not in ("egocentric", "legacy"):
            raise ValueError(
                f"observation_mode must be 'egocentric' or 'legacy', got {observation_mode!r}"
            )
        self.observation_frame = str(observation_frame).lower()
        if self.observation_frame not in ("world", "goal"):
            raise ValueError(
                f"observation_frame must be 'world' or 'goal', got {observation_frame!r}"
            )
        self.n_lidar_rays = int(n_lidar_rays)
        self.lidar_range = (
            float(lidar_range) if lidar_range is not None else self.grid_size * 0.75
        )
        self.max_sheep = int(max_sheep) if max_sheep is not None else self.base_n_sheep
        if self.max_sheep < self.base_n_sheep:
            raise ValueError("max_sheep must be >= n_sheep")
        if self.observation_mode == "legacy" and self.randomize_sheep_count:
            raise ValueError(
                "randomize_sheep_count requires observation_mode='egocentric'; the "
                "legacy layout has one fixed slot per sheep."
            )
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_seen_centroid: Optional[np.ndarray] = None
        self._steps_since_seen = 0
        self._fraction_at_goal = 0.0
        self._best_fraction_at_goal = 0.0
        self._steps_to_80pct = -1
        self.observation_noise = 0.0
        self.action_noise = 0.0
        self.topology = "preset"

        self.goal = self.base_goal.copy()
        self.visibility_radius = self.base_visibility_radius
        self.sheep_speed = self.base_sheep_speed
        self.cohesion_factor = self.base_cohesion_factor
        self.repulsion_strength = self.base_repulsion_strength
        self.leader_factor = self.base_leader_factor
        self.obstacle_avoidance_threshold = self.base_obstacle_avoidance_threshold
        self.obstacles: List[Rect] = []
        self.spawn_bounds = default_spawn_bounds()

        if self.observation_mode == "legacy":
            obs_dim = 4 + 2 * self.n_sheep + 4 * self.max_obstacles
            self.observation_space = spaces.Box(
                low=-self.grid_size,
                high=_SENTINEL + 1.0,
                shape=(obs_dim,),
                dtype=np.float32,
            )
        else:
            obs_dim = _EGO_SCALAR_BLOCK + 3 * self.max_sheep + self.n_lidar_rays
            self.observation_space = spaces.Box(
                low=-_EGO_OBS_BOUND,
                high=_EGO_OBS_BOUND,
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
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._last_seen_centroid = None
        self._steps_since_seen = 0
        self._fraction_at_goal = 0.0
        self._best_fraction_at_goal = 0.0
        self._steps_to_80pct = -1
        initial_vis_mask = visible_sheep_mask(
            self.dog_pos, self.sheep_pos, self.visibility_radius
        )
        self._update_flock_memory(initial_vis_mask)
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
        # The policy acts in the same frame it observes in, so a goal-aligned
        # observation implies a goal-aligned action. The basis is taken before the
        # dog moves, matching the observation the policy was given.
        if self.observation_frame == "goal" and self.observation_mode != "legacy":
            action = self._to_world(action)
        if self.action_noise > 0.0:
            action = action + self._rng.normal(0.0, self.action_noise, size=2).astype(
                np.float32
            )
            action_norm = float(np.linalg.norm(action))
            if action_norm > 1.0:
                action = action / action_norm
        self._prev_action = action.astype(np.float32)

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
        self._track_delivery(dists_to_goal)
        mean_dist_to_goal = float(np.mean(dists_to_goal))
        max_dist_to_goal = float(np.max(dists_to_goal))
        progress_delta = self._prev_mean_dist_to_goal - mean_dist_to_goal
        progress_reward = self.progress_reward_scale * progress_delta
        worst_sheep_delta = self._prev_max_dist_to_goal - max_dist_to_goal
        worst_sheep_reward = self.worst_sheep_reward_scale * worst_sheep_delta

        vis_mask = visible_sheep_mask(self.dog_pos, self.sheep_pos, self.visibility_radius)
        visible_ratio = float(np.mean(vis_mask))
        self._update_flock_memory(vis_mask)
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
            success_bonus = self.success_bonus
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
        if scenario_name in ("train", "validation", "test_procedural"):
            return self._sample_procedural_scenario(scenario_name)
        return self._sample_preset_scenario(scenario_name)

    def _stage_randomization(self, scenario_name: str) -> Dict[str, Any]:
        """Return the randomization breadth for the current curriculum stage.

        The stage no longer switches whole randomization axes on and off; it
        widens them. Obstacles in particular are always present in some form, so
        an agent that stalls at an early stage still sees structure rather than a
        single fixed layout.
        """
        if scenario_name == "validation":
            return {
                "layout_spec": TRAIN_LAYOUT_SPEC,
                "dynamics_spec": TRAIN_DYNAMICS_SPEC,
                "breadth": 1.0,
                "randomize_goal": True,
                "randomize_sheep_count": self.randomize_sheep_count,
            }
        if scenario_name == "test_procedural":
            return {
                "layout_spec": TEST_LAYOUT_SPEC,
                "dynamics_spec": TEST_DYNAMICS_SPEC,
                "breadth": 1.0,
                "randomize_goal": True,
                "randomize_sheep_count": self.randomize_sheep_count,
            }

        stage = float(self.curriculum_stage) if self.curriculum_mode else 1.0
        if stage < 0.33:
            weights = {"open": 0.35, "blobs": 0.45, "bars": 0.20}
            breadth = 0.25
            randomize_goal = False
        elif stage < 0.66:
            weights = {"open": 0.18, "blobs": 0.32, "bars": 0.25, "gate": 0.25}
            breadth = 0.6
            randomize_goal = self.randomize_goal
        else:
            weights = dict(TRAIN_LAYOUT_SPEC.topology_weights)
            breadth = 1.0
            randomize_goal = self.randomize_goal

        layout_spec = replace(TRAIN_LAYOUT_SPEC, topology_weights=weights)
        if not self.randomize_obstacles:
            layout_spec = replace(layout_spec, topology_weights={"blobs": 1.0})
        return {
            "layout_spec": layout_spec,
            "dynamics_spec": TRAIN_DYNAMICS_SPEC,
            "breadth": breadth,
            "randomize_goal": randomize_goal,
            "randomize_sheep_count": self.randomize_sheep_count and stage >= 0.66,
        }

    def _lerp_range(
        self, base: float, span: Tuple[float, float], breadth: float
    ) -> Tuple[float, float]:
        """Shrink *span* toward *base* as the curriculum breadth decreases."""
        low = base + (float(span[0]) - base) * breadth
        high = base + (float(span[1]) - base) * breadth
        return (min(low, high), max(low, high))

    def _sample_procedural_scenario(self, scenario_name: str) -> ScenarioConfig:
        settings = self._stage_randomization(scenario_name)
        layout_spec = settings["layout_spec"]
        dynamics: DynamicsSpec = settings["dynamics_spec"]
        breadth = float(settings["breadth"])

        if not self.domain_randomization:
            obstacles = tuple(fixed_training_obstacles(self.grid_size))
            if self.structured_train_obstacles:
                layouts = structured_training_obstacle_layouts(self.grid_size)
                obstacles = tuple(layouts[int(self._rng.integers(0, len(layouts)))])
            goal = tuple(float(v) for v in self.base_goal)
            return ScenarioConfig(
                goal=goal,
                visibility_radius=self.base_visibility_radius,
                obstacles=obstacles,
                sheep_speed=self.base_sheep_speed,
                cohesion_factor=self.base_cohesion_factor,
                repulsion_strength=self.base_repulsion_strength,
                leader_factor=self.base_leader_factor,
                obstacle_avoidance_threshold=self.base_obstacle_avoidance_threshold,
                spawn_bounds=(
                    opposite_goal_spawn_bounds(goal, self.grid_size)
                    if self.opposite_goal_spawn
                    else default_spawn_bounds()
                ),
                n_sheep=self.base_n_sheep,
                flee_radius=self.base_flee_radius,
                topology="fixed",
            )

        # Goal and spawn are drawn independently so the policy cannot rely on a
        # fixed goal direction.
        if settings["randomize_goal"]:
            goal, spawn_bounds = sample_goal_and_spawn(self._rng, self.grid_size)
        else:
            goal = tuple(float(v) for v in self.base_goal)
            spawn_bounds = (
                opposite_goal_spawn_bounds(goal, self.grid_size)
                if self.opposite_goal_spawn
                else default_spawn_bounds()
            )

        spawn_centre = (
            float((spawn_bounds[0][0] + spawn_bounds[1][0]) * 0.5 * self.grid_size),
            float((spawn_bounds[0][1] + spawn_bounds[1][1]) * 0.5 * self.grid_size),
        )
        topology, obstacle_list = sample_layout(
            self._rng,
            self.grid_size,
            layout_spec,
            keepouts=[spawn_centre, goal],
            keepout_radius=max(self.success_radius * 1.3, 2.0),
            max_obstacles=self.max_obstacles,
        )
        obstacles = tuple(obstacle_list)

        visibility_radius = self.base_visibility_radius
        if self.randomize_visibility:
            span = self._lerp_range(
                self.base_visibility_radius, dynamics.visibility_radius, breadth
            )
            visibility_radius = float(self._rng.uniform(*span))

        sheep_speed = self.base_sheep_speed
        cohesion_factor = self.base_cohesion_factor
        repulsion_strength = self.base_repulsion_strength
        leader_factor = self.base_leader_factor
        obstacle_threshold = self.base_obstacle_avoidance_threshold
        flee_radius = self.base_flee_radius
        observation_noise = 0.0
        action_noise = 0.0
        if self.randomize_dynamics:
            sheep_speed = float(
                self._rng.uniform(
                    *self._lerp_range(self.base_sheep_speed, dynamics.sheep_speed, breadth)
                )
            )
            cohesion_factor = float(
                self._rng.uniform(
                    *self._lerp_range(
                        self.base_cohesion_factor, dynamics.cohesion_factor, breadth
                    )
                )
            )
            repulsion_strength = float(
                self._rng.uniform(
                    *self._lerp_range(
                        self.base_repulsion_strength, dynamics.repulsion_strength, breadth
                    )
                )
            )
            leader_factor = float(
                self._rng.uniform(
                    *self._lerp_range(
                        self.base_leader_factor, dynamics.leader_factor, breadth
                    )
                )
            )
            obstacle_threshold = float(
                self._rng.uniform(
                    *self._lerp_range(
                        self.base_obstacle_avoidance_threshold,
                        dynamics.obstacle_avoidance_threshold,
                        breadth,
                    )
                )
            )
            flee_radius = float(
                self._rng.uniform(
                    *self._lerp_range(self.base_flee_radius, dynamics.flee_radius, breadth)
                )
            )
            observation_noise = float(
                self._rng.uniform(*dynamics.observation_noise) * breadth
            )
            action_noise = float(self._rng.uniform(*dynamics.action_noise) * breadth)

        n_sheep = self.base_n_sheep
        if settings["randomize_sheep_count"]:
            low = max(int(dynamics.n_sheep[0]), 1)
            high = min(int(dynamics.n_sheep[1]), self.max_sheep)
            if high >= low:
                n_sheep = int(self._rng.integers(low, high + 1))

        return ScenarioConfig(
            goal=goal,
            visibility_radius=visibility_radius,
            obstacles=obstacles,
            sheep_speed=sheep_speed,
            cohesion_factor=cohesion_factor,
            repulsion_strength=repulsion_strength,
            leader_factor=leader_factor,
            obstacle_avoidance_threshold=obstacle_threshold,
            visibility_radius_range=dynamics.visibility_radius,
            spawn_bounds=spawn_bounds,
            n_sheep=n_sheep,
            flee_radius=flee_radius,
            observation_noise=observation_noise,
            action_noise=action_noise,
            topology=topology,
        )

    def _sample_preset_scenario(self, scenario_name: str) -> ScenarioConfig:
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
            n_sheep=self.base_n_sheep,
            flee_radius=self.base_flee_radius,
            topology="preset",
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
        self.n_sheep = int(config.n_sheep if config.n_sheep is not None else self.base_n_sheep)
        self.flee_radius = float(
            config.flee_radius if config.flee_radius is not None else self.base_flee_radius
        )
        self.observation_noise = float(config.observation_noise)
        self.action_noise = float(config.action_noise)
        self.topology = str(config.topology)

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

    def _track_delivery(self, dists_to_goal: np.ndarray) -> None:
        """Track graded delivery progress alongside the all-or-nothing success flag.

        ``all_at_goal`` requires every sheep inside the goal radius, so a single
        permanently-lost sheep zeroes an otherwise good episode. These signals let
        an evaluation distinguish "delivered nine of ten" from "delivered none".
        """
        fraction = float(np.mean(dists_to_goal < self.success_radius))
        self._fraction_at_goal = fraction
        self._best_fraction_at_goal = max(self._best_fraction_at_goal, fraction)
        if self._steps_to_80pct < 0 and fraction >= 0.8:
            self._steps_to_80pct = int(self.current_step)

    def _update_flock_memory(self, vis_mask: np.ndarray) -> None:
        """Remember where the flock was last seen, and for how long it has been lost."""
        if np.any(vis_mask):
            self._last_seen_centroid = np.mean(
                self.sheep_pos[np.flatnonzero(vis_mask)], axis=0
            ).astype(np.float32)
            self._steps_since_seen = 0
        else:
            self._steps_since_seen += 1

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

    def _frame_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the orthonormal basis (e1, e2) the observation is expressed in.

        With ``observation_frame='goal'`` the first axis points at the goal, which
        makes the whole observation (and, correspondingly, the action) invariant
        to rotations of the task. That removes the "goal is always up-and-right"
        shortcut that the training goal distribution would otherwise hand the
        policy.
        """
        if self.observation_frame == "world":
            return (
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            )
        goal_vec = self.goal - self.dog_pos
        norm = float(np.linalg.norm(goal_vec))
        if norm < 1e-8:
            e1 = np.array([1.0, 0.0], dtype=np.float32)
        else:
            e1 = (goal_vec / norm).astype(np.float32)
        e2 = np.array([-e1[1], e1[0]], dtype=np.float32)
        return e1, e2

    def _to_frame(self, vectors: np.ndarray) -> np.ndarray:
        """Project world-frame vectors onto the observation frame."""
        e1, e2 = self._frame_basis()
        basis = np.stack([e1, e2], axis=1).astype(np.float32)
        return (np.atleast_2d(vectors).astype(np.float32) @ basis).astype(np.float32)

    def _to_world(self, vector: np.ndarray) -> np.ndarray:
        """Map a vector expressed in the observation frame back to world axes."""
        e1, e2 = self._frame_basis()
        return (float(vector[0]) * e1 + float(vector[1]) * e2).astype(np.float32)

    def _boundary_clearances(self, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        """Distance from the dog to the arena boundary along +-e1 and +-e2."""
        clearances = []
        for direction in (e1, -e1, e2, -e2):
            distance = float(self.grid_size) * 2.0
            for axis in (0, 1):
                component = float(direction[axis])
                if abs(component) <= 1e-9:
                    continue
                bound = self.grid_size if component > 0.0 else 0.0
                distance = min(distance, (bound - float(self.dog_pos[axis])) / component)
            clearances.append(max(distance, 0.0))
        return np.asarray(clearances, dtype=np.float32)

    def _get_obs(self, vis_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if vis_mask is None:
            vis_mask = visible_sheep_mask(
                self.dog_pos, self.sheep_pos, self.visibility_radius
            )
        if self.observation_mode == "legacy":
            return self._get_legacy_obs(vis_mask)
        return self._get_egocentric_obs(vis_mask)

    def _get_egocentric_obs(self, vis_mask: np.ndarray) -> np.ndarray:
        """Normalized, masked, ray-based observation.

        Every feature is scaled into roughly ``[-2, 2]``: unseen sheep are encoded
        as zeros plus a visibility flag rather than an out-of-range sentinel, the
        dog's absolute position is replaced by boundary clearances, and obstacles
        are replaced by an egocentric lidar sweep.
        """
        e1, e2 = self._frame_basis()
        radius = max(self.visibility_radius, 1e-6)
        diag = float(np.sqrt(2.0) * self.grid_size)

        goal_vec = self.goal - self.dog_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        goal_frame = self._to_frame(goal_vec)[0]
        goal_unit = goal_frame / goal_dist if goal_dist > 1e-8 else np.zeros(2, np.float32)
        goal_block = np.array(
            [
                goal_unit[0],
                goal_unit[1],
                min(goal_dist / diag, 1.0),
                min(goal_dist / radius, 2.0) * 0.5,
            ],
            dtype=np.float32,
        )

        boundary_block = (self._boundary_clearances(e1, e2) / diag).astype(np.float32)
        action_block = self._to_frame(self._prev_action)[0]

        sheep_block = np.zeros(3 * self.max_sheep, dtype=np.float32)
        visible_idx = np.flatnonzero(vis_mask)
        # Sensor noise is applied to the perceived positions only; the underlying
        # simulation state is untouched.
        perceived = self.sheep_pos
        if self.observation_noise > 0.0 and visible_idx.size > 0:
            perceived = self.sheep_pos + self._rng.normal(
                0.0, self.observation_noise, size=self.sheep_pos.shape
            ).astype(np.float32)
        if visible_idx.size > 0:
            rel = self._to_frame(perceived[visible_idx] - self.dog_pos) / radius
            sheep_block[3 * visible_idx] = rel[:, 0]
            sheep_block[3 * visible_idx + 1] = rel[:, 1]
            sheep_block[3 * visible_idx + 2] = 1.0

        summary_block = np.zeros(8, dtype=np.float32)
        if visible_idx.size > 0:
            rel_world = perceived[visible_idx] - self.dog_pos
            rel_frame = self._to_frame(rel_world)
            dists = np.linalg.norm(rel_world, axis=1)
            centroid_frame = np.mean(rel_frame, axis=0)
            summary_block = np.array(
                [
                    float(np.mean(vis_mask)),
                    float(visible_idx.size) / float(self.max_sheep),
                    centroid_frame[0] / radius,
                    centroid_frame[1] / radius,
                    float(np.mean(dists)) / radius,
                    float(np.max(dists)) / radius,
                    float(np.std(rel_frame[:, 0])) / radius,
                    float(np.std(rel_frame[:, 1])) / radius,
                ],
                dtype=np.float32,
            )

        memory_block = np.zeros(4, dtype=np.float32)
        centroid_goal_block = np.zeros(3, dtype=np.float32)
        if self._last_seen_centroid is not None:
            remembered = self._to_frame(self._last_seen_centroid - self.dog_pos)[0] / radius
            memory_block = np.array(
                [
                    remembered[0],
                    remembered[1],
                    min(float(self._steps_since_seen) / _MEMORY_HORIZON, 1.0),
                    1.0,
                ],
                dtype=np.float32,
            )
            to_goal = self.goal - self._last_seen_centroid
            to_goal_dist = float(np.linalg.norm(to_goal))
            to_goal_frame = self._to_frame(to_goal)[0]
            if to_goal_dist > 1e-8:
                to_goal_frame = to_goal_frame / to_goal_dist
            centroid_goal_block = np.array(
                [to_goal_frame[0], to_goal_frame[1], min(to_goal_dist / diag, 1.0)],
                dtype=np.float32,
            )

        angles = ray_angles(self.n_lidar_rays, offset=float(np.arctan2(e1[1], e1[0])))
        lidar_block = (
            raycast_distances(
                self.dog_pos,
                angles,
                self.obstacles,
                self.grid_size,
                self.lidar_range,
            )
            / self.lidar_range
        ).astype(np.float32)

        time_block = np.array(
            [1.0 - min(float(self.current_step) / float(self.max_steps), 1.0)],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                goal_block,
                boundary_block,
                action_block,
                summary_block,
                memory_block,
                centroid_goal_block,
                time_block,
                sheep_block,
                lidar_block,
            ]
        ).astype(np.float32)
        return np.clip(obs, -_EGO_OBS_BOUND, _EGO_OBS_BOUND)

    def _get_legacy_obs(self, vis_mask: np.ndarray) -> np.ndarray:
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
            "topology": self.topology,
            "mean_dist_to_goal": float(np.mean(dists_to_goal)),
            "max_dist_to_goal": float(np.max(dists_to_goal)),
            "all_at_goal": bool(np.all(dists_to_goal < self.success_radius)),
            "fraction_at_goal": float(np.mean(dists_to_goal < self.success_radius)),
            "best_fraction_at_goal": float(self._best_fraction_at_goal),
            "steps_to_80pct_collected": int(self._steps_to_80pct),
            "n_sheep": int(self.n_sheep),
            "visibility_radius": float(self.visibility_radius),
            "n_obstacles": int(len(self.obstacles)),
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
