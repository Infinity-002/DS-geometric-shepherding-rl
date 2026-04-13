"""Feature engineering helpers for imitation learning."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from shepherding.utils.geometry import compute_convex_hull


def observation_to_features(
    observation: np.ndarray,
    n_sheep: int,
    *,
    sentinel: float = 999.0,
) -> np.ndarray:
    """Convert one environment observation into a fixed-size feature vector."""
    obs = np.asarray(observation, dtype=np.float32).flatten()
    dog_pos = obs[:2]
    goal_offset = obs[2:4]
    sheep_end = 4 + (2 * n_sheep)
    sheep_rel = obs[4:sheep_end].reshape(n_sheep, 2)
    obstacle_flat = obs[sheep_end:]
    visible_mask = sheep_rel[:, 0] < (sentinel * 0.5)

    raw_sheep = sheep_rel.copy()
    raw_sheep[~visible_mask] = 0.0

    if np.any(visible_mask):
        visible_rel = sheep_rel[visible_mask]
        visible_abs = dog_pos + visible_rel
        centroid_rel = np.mean(visible_rel, axis=0)
        centroid_abs = dog_pos + centroid_rel
        dists_to_centroid = np.linalg.norm(visible_abs - centroid_abs, axis=1)
        dists_to_goal = np.linalg.norm(visible_abs - (dog_pos + goal_offset), axis=1)
        dists_to_dog = np.linalg.norm(visible_rel, axis=1)
        hull = compute_convex_hull(visible_abs)
        hull_area = float(hull.volume) if hull is not None else _bounding_box_area(visible_abs)
        focus_cluster = _focus_cluster(visible_abs, dog_pos + goal_offset)
        cluster_centroid = np.mean(focus_cluster, axis=0)
        cluster_goal_dist = float(np.linalg.norm(cluster_centroid - (dog_pos + goal_offset)))
        focus_fraction = float(focus_cluster.shape[0]) / float(visible_abs.shape[0])
        summary = np.array(
            [
                float(np.mean(visible_mask)),
                float(visible_abs.shape[0]),
                float(centroid_rel[0]),
                float(centroid_rel[1]),
                float(np.mean(dists_to_dog)),
                float(np.max(dists_to_dog)),
                float(np.mean(dists_to_goal)),
                float(np.max(dists_to_goal)),
                float(np.mean(dists_to_centroid)),
                float(np.max(dists_to_centroid)),
                hull_area,
                float(np.std(visible_abs[:, 0])),
                float(np.std(visible_abs[:, 1])),
                cluster_goal_dist,
                focus_fraction,
            ],
            dtype=np.float32,
        )
    else:
        summary = np.zeros(15, dtype=np.float32)

    return np.concatenate(
        [
            dog_pos.astype(np.float32),
            goal_offset.astype(np.float32),
            raw_sheep.reshape(-1).astype(np.float32),
            obstacle_flat.astype(np.float32),
            summary,
        ]
    ).astype(np.float32)


@lru_cache(maxsize=32)
def feature_names(n_sheep: int, max_obstacles: int) -> list[str]:
    names = ["dog_x", "dog_y", "goal_dx", "goal_dy"]
    for sheep_idx in range(n_sheep):
        names.extend([f"sheep_{sheep_idx}_dx", f"sheep_{sheep_idx}_dy"])
    for obstacle_idx in range(max_obstacles):
        names.extend(
            [
                f"obstacle_{obstacle_idx}_x",
                f"obstacle_{obstacle_idx}_y",
                f"obstacle_{obstacle_idx}_w",
                f"obstacle_{obstacle_idx}_h",
            ]
        )
    names.extend(
        [
            "visible_ratio",
            "visible_count",
            "centroid_dx",
            "centroid_dy",
            "mean_sheep_dog_dist",
            "max_sheep_dog_dist",
            "mean_sheep_goal_dist",
            "max_sheep_goal_dist",
            "mean_sheep_centroid_dist",
            "max_sheep_centroid_dist",
            "visible_hull_area",
            "visible_std_x",
            "visible_std_y",
            "focus_cluster_goal_dist",
            "focus_cluster_fraction",
        ]
    )
    return names


def _bounding_box_area(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 0.0
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return float(np.prod(maxs - mins))


def _focus_cluster(points: np.ndarray, goal: np.ndarray) -> np.ndarray:
    if points.shape[0] < 4:
        return points
    first, second = _farthest_pair(points)
    if first == second:
        return points
    seed_a = points[first]
    seed_b = points[second]
    if float(np.linalg.norm(seed_a - seed_b)) < 2.0:
        return points
    dist_a = np.linalg.norm(points - seed_a, axis=1)
    dist_b = np.linalg.norm(points - seed_b, axis=1)
    mask_a = dist_a <= dist_b
    mask_b = ~mask_a
    if not np.any(mask_a) or not np.any(mask_b):
        return points
    cluster_a = points[mask_a]
    cluster_b = points[mask_b]
    goal_dist_a = float(np.linalg.norm(np.mean(cluster_a, axis=0) - goal))
    goal_dist_b = float(np.linalg.norm(np.mean(cluster_b, axis=0) - goal))
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
