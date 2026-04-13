"""Heuristic baseline controller for shepherding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from shepherding.utils.geometry_v2 import obstacle_avoidance_force


@dataclass
class HeuristicShepherdAgent:
    """Simple collect-and-drive baseline using visible sheep only."""

    n_sheep: int
    max_obstacles: int
    grid_size: float
    visibility_radius: float
    flee_radius: float
    success_radius: float
    sentinel: float = 999.0
    collect_distance_scale: float = 1.6
    drive_distance_scale: float = 0.85
    obstacle_threshold: float = 1.25
    use_cluster_targets: bool = False
    cluster_activation_distance: float = 2.0

    def __post_init__(self) -> None:
        self._last_seen_centroid: Optional[np.ndarray] = None
        self._search_sign = 1.0

    def reset(self) -> None:
        self._last_seen_centroid = None
        self._search_sign = 1.0

    def predict(
        self,
        observation: np.ndarray,
        state: object | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        if episode_start is not None and np.any(episode_start):
            self.reset()

        obs = np.asarray(observation, dtype=np.float32)
        dog_pos = obs[:2]
        goal = dog_pos + obs[2:4]
        sheep_flat = obs[4 : 4 + 2 * self.n_sheep]
        obstacle_flat = obs[4 + 2 * self.n_sheep :]

        visible_mask = sheep_flat[0::2] < (self.sentinel * 0.5)
        visible_rel = sheep_flat.reshape(self.n_sheep, 2)[visible_mask]
        obstacles = self._decode_obstacles(obstacle_flat)

        if visible_rel.size == 0:
            action = self._search_action(dog_pos, goal, obstacles)
            return action.astype(np.float32), None

        sheep_pos = dog_pos + visible_rel
        cluster_pos = sheep_pos
        if self.use_cluster_targets and sheep_pos.shape[0] >= 4:
            cluster_pos = _select_focus_cluster(
                sheep_pos,
                goal,
                min_separation=self.cluster_activation_distance,
            )

        centroid = np.mean(cluster_pos, axis=0)
        self._last_seen_centroid = centroid

        offsets = cluster_pos - centroid
        dists = np.linalg.norm(offsets, axis=1)
        furthest_idx = int(np.argmax(dists))
        furthest = cluster_pos[furthest_idx]
        goal_dir = _safe_unit(goal - centroid, fallback=np.array([1.0, 0.0], dtype=np.float32))

        if float(dists[furthest_idx]) > self.success_radius * self.collect_distance_scale:
            collect_dir = _safe_unit(furthest - centroid, fallback=-goal_dir)
            target = furthest + collect_dir * min(self.flee_radius * 0.75, self.visibility_radius * 0.55)
        else:
            target = centroid - goal_dir * min(
                self.flee_radius * self.drive_distance_scale,
                self.visibility_radius * 0.8,
            )

        steer = target - dog_pos
        if obstacles:
            steer = steer + 0.9 * obstacle_avoidance_force(
                dog_pos,
                obstacles,
                threshold=self.obstacle_threshold,
            )
        return _safe_unit(steer).astype(np.float32), None

    def _search_action(
        self,
        dog_pos: np.ndarray,
        goal: np.ndarray,
        obstacles: list[tuple[float, float, float, float]],
    ) -> np.ndarray:
        if self._last_seen_centroid is not None:
            search_target = self._last_seen_centroid
        else:
            goal_dir = _safe_unit(goal - dog_pos, fallback=np.array([1.0, 0.0], dtype=np.float32))
            lateral = np.array([-goal_dir[1], goal_dir[0]], dtype=np.float32) * self._search_sign
            self._search_sign *= -1.0
            search_target = dog_pos + goal_dir * (self.visibility_radius * 0.35) + lateral

        steer = search_target - dog_pos
        if obstacles:
            steer = steer + 0.9 * obstacle_avoidance_force(
                dog_pos,
                obstacles,
                threshold=self.obstacle_threshold,
            )
        return _safe_unit(steer)

    def _decode_obstacles(self, obstacle_flat: np.ndarray) -> list[tuple[float, float, float, float]]:
        obstacles: list[tuple[float, float, float, float]] = []
        for idx in range(self.max_obstacles):
            base = 4 * idx
            rx, ry, rw, rh = obstacle_flat[base : base + 4]
            if rx < 0.0:
                continue
            obstacles.append(
                (
                    float(rx * self.grid_size),
                    float(ry * self.grid_size),
                    float(rw * self.grid_size),
                    float(rh * self.grid_size),
                )
            )
        return obstacles


def _safe_unit(vec: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        return vec / norm
    if fallback is not None:
        return _safe_unit(np.asarray(fallback, dtype=np.float32))
    return np.zeros(2, dtype=np.float32)


def _select_focus_cluster(
    sheep_pos: np.ndarray,
    goal: np.ndarray,
    min_separation: float,
) -> np.ndarray:
    """Pick the cluster farthest from the goal when the flock clearly splits."""
    first, second = _farthest_pair(sheep_pos)
    if first == second:
        return sheep_pos

    seed_a = sheep_pos[first]
    seed_b = sheep_pos[second]
    if float(np.linalg.norm(seed_a - seed_b)) < min_separation:
        return sheep_pos

    dist_a = np.linalg.norm(sheep_pos - seed_a, axis=1)
    dist_b = np.linalg.norm(sheep_pos - seed_b, axis=1)
    assign_a = dist_a <= dist_b
    assign_b = ~assign_a
    if not np.any(assign_a) or not np.any(assign_b):
        return sheep_pos

    cluster_a = sheep_pos[assign_a]
    cluster_b = sheep_pos[assign_b]
    centroid_a = np.mean(cluster_a, axis=0)
    centroid_b = np.mean(cluster_b, axis=0)
    goal_dist_a = float(np.linalg.norm(centroid_a - goal))
    goal_dist_b = float(np.linalg.norm(centroid_b - goal))
    return cluster_a if goal_dist_a >= goal_dist_b else cluster_b


def _farthest_pair(points: np.ndarray) -> tuple[int, int]:
    max_dist = -1.0
    best = (0, 0)
    for i in range(points.shape[0]):
        deltas = points[i + 1 :] - points[i]
        if deltas.size == 0:
            continue
        dists = np.linalg.norm(deltas, axis=1)
        idx = int(np.argmax(dists))
        if float(dists[idx]) > max_dist:
            max_dist = float(dists[idx])
            best = (i, i + 1 + idx)
    return best
