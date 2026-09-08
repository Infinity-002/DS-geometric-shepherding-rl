"""
geometry_v2.py – Geometry utilities for the enhanced (v2) shepherding environment.

Extends geometry.py with obstacle-aware helpers:
  * point_in_rect         – axis-aligned rectangle containment test
  * clip_to_free_space    – prevents a movement step from entering any obstacle
  * obstacle_avoidance_force – smooth push-away force near obstacle edges
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# Obstacle defined as (x_min, y_min, width, height)
Rect = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Obstacle helpers
# ---------------------------------------------------------------------------

def point_in_rect(p: np.ndarray, rect: Rect) -> bool:
    """Return True if point *p* lies strictly inside axis-aligned *rect*.

    Parameters
    ----------
    p : np.ndarray
        Shape ``(2,)`` – the query point ``[x, y]``.
    rect : tuple
        ``(x_min, y_min, width, height)`` rectangle.
    """
    x, y = float(p[0]), float(p[1])
    rx, ry, rw, rh = rect
    return (rx < x < rx + rw) and (ry < y < ry + rh)


def segment_intersects_rect(start: np.ndarray, end: np.ndarray, rect: Rect) -> bool:
    """Return True if the closed segment from *start* to *end* hits *rect*."""
    if point_in_rect(start, rect) or point_in_rect(end, rect):
        return True

    rx, ry, rw, rh = rect
    x_min, x_max = rx, rx + rw
    y_min, y_max = ry, ry + rh
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    t0 = 0.0
    t1 = 1.0

    for p, q in (
        (-dx, float(start[0] - x_min)),
        (dx, float(x_max - start[0])),
        (-dy, float(start[1] - y_min)),
        (dy, float(y_max - start[1])),
    ):
        if abs(p) <= 1e-12:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False

    return True


def clip_to_free_space(
    current: np.ndarray,
    proposed: np.ndarray,
    obstacles: List[Rect],
    grid_size: float,
) -> np.ndarray:
    """Clamp movement to the furthest point along the step that remains free.

    Unlike the original implementation, this also catches segment crossings,
    so agents cannot tunnel through thin obstacles in a single step.

    Parameters
    ----------
    current : np.ndarray
        Shape ``(2,)`` – current position (guaranteed free).
    proposed : np.ndarray
        Shape ``(2,)`` – candidate new position after applying velocity.
    obstacles : list of Rect
        Obstacle rectangles ``(x_min, y_min, w, h)``.
    grid_size : float
        World boundary – positions are clamped to ``[0, grid_size]``.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` – collision-resolved position clamped to the grid.
    """
    start = np.clip(current, 0.0, grid_size).astype(np.float32)
    end = np.clip(proposed, 0.0, grid_size).astype(np.float32)

    if np.allclose(start, end):
        return end

    def is_free(point: np.ndarray) -> bool:
        return not any(point_in_rect(point, rect) for rect in obstacles)

    def is_reachable(point: np.ndarray) -> bool:
        return is_free(point) and not any(
            segment_intersects_rect(start, point, rect) for rect in obstacles
        )

    if is_reachable(end):
        return end

    low = 0.0
    high = 1.0
    for _ in range(24):
        mid = (low + high) / 2.0
        point = start + (end - start) * mid
        if is_reachable(point):
            low = mid
        else:
            high = mid

    clipped = start + (end - start) * low
    return np.clip(clipped, 0.0, grid_size).astype(np.float32)


def obstacle_avoidance_force(
    pos: np.ndarray,
    obstacles: List[Rect],
    threshold: float = 1.5,
) -> np.ndarray:
    """Compute a smooth repulsion force pushing *pos* away from nearby obstacles.

    The force linearly decays from ``threshold`` distance to zero.

    Parameters
    ----------
    pos : np.ndarray
        Shape ``(2,)`` – agent position.
    obstacles : list of Rect
        Obstacle rectangles ``(x_min, y_min, w, h)``.
    threshold : float
        Distance (from closest rectangle point) at which avoidance kicks in.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` – avoidance velocity contribution.
    """
    force = np.zeros(2, dtype=np.float32)
    for rect in obstacles:
        rx, ry, rw, rh = rect
        # Closest point on rect boundary to pos
        cx = float(np.clip(pos[0], rx, rx + rw))
        cy = float(np.clip(pos[1], ry, ry + rh))
        diff = pos - np.array([cx, cy], dtype=np.float32)
        dist = float(np.linalg.norm(diff))
        if dist < threshold and dist > 1e-8:
            strength = (threshold - dist) / threshold
            force += (diff / dist) * strength
    return force


def obstacle_avoidance_forces(
    positions: np.ndarray,
    obstacles: List[Rect],
    threshold: float = 1.5,
) -> np.ndarray:
    """Vectorized obstacle avoidance for many positions at once."""
    if len(obstacles) == 0:
        return np.zeros_like(positions, dtype=np.float32)

    obstacle_array = np.asarray(obstacles, dtype=np.float32)
    mins = obstacle_array[:, :2]
    maxs = mins + obstacle_array[:, 2:]

    clipped = np.clip(positions[:, None, :], mins[None, :, :], maxs[None, :, :])
    diffs = positions[:, None, :] - clipped
    dists_sq = np.sum(diffs * diffs, axis=2)

    threshold_sq = float(threshold * threshold)
    mask = (dists_sq < threshold_sq) & (dists_sq > 1e-12)
    dists = np.zeros_like(dists_sq, dtype=np.float32)
    dists[mask] = np.sqrt(dists_sq[mask]).astype(np.float32)

    strengths = np.zeros_like(dists_sq, dtype=np.float32)
    strengths[mask] = (threshold - dists[mask]) / threshold

    unit = np.zeros_like(diffs, dtype=np.float32)
    unit[mask] = diffs[mask] / dists[mask, None]
    return np.sum(unit * strengths[..., None], axis=1, dtype=np.float32)


# ---------------------------------------------------------------------------
# Observation masking
# ---------------------------------------------------------------------------

def visible_sheep_mask(
    dog_pos: np.ndarray,
    sheep_pos: np.ndarray,
    visibility_radius: float,
) -> np.ndarray:
    """Return a boolean mask of shape ``(N,)`` – True where sheep are visible.

    Parameters
    ----------
    dog_pos : np.ndarray
        Shape ``(2,)`` – dog position.
    sheep_pos : np.ndarray
        Shape ``(N, 2)`` – sheep positions.
    visibility_radius : float
        Maximum distance at which the dog can perceive a sheep.

    Returns
    -------
    np.ndarray
        Boolean array, True for visible sheep.
    """
    diffs = sheep_pos - dog_pos
    dists_sq = np.sum(diffs * diffs, axis=1)
    return dists_sq <= float(visibility_radius * visibility_radius)


# ---------------------------------------------------------------------------
# Egocentric ray casting
# ---------------------------------------------------------------------------

def raycast_distances(
    origin: np.ndarray,
    angles: np.ndarray,
    obstacles: Sequence[Rect],
    grid_size: float,
    max_range: float,
) -> np.ndarray:
    """Cast rays from *origin* and return the free distance along each one.

    Rays are occluded by the axis-aligned obstacle rectangles *and* by the arena
    boundary, so the resulting vector is a complete, egocentric description of
    the local free space. Unlike an indexed list of obstacle rectangles this
    representation is invariant to obstacle count, ordering, absolute position
    and shape, which is what lets a policy trained on scattered blobs transfer to
    corridors and gates.

    Parameters
    ----------
    origin : np.ndarray
        Shape ``(2,)`` – ray origin, assumed to lie in free space.
    angles : np.ndarray
        Shape ``(R,)`` – ray directions in radians.
    obstacles : sequence of Rect
        Obstacle rectangles ``(x_min, y_min, w, h)``.
    grid_size : float
        Arena side length; the arena boundary occludes rays.
    max_range : float
        Rays are truncated at this distance.

    Returns
    -------
    np.ndarray
        Shape ``(R,)`` – distance to the first occluder along each ray, clipped
        to ``[0, max_range]``.
    """
    origin = np.asarray(origin, dtype=np.float32).reshape(2)
    angles = np.asarray(angles, dtype=np.float32).reshape(-1)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    max_range = float(max_range)
    hits = np.full(angles.shape[0], max_range, dtype=np.float32)

    # The arena boundary behaves like a rectangle seen from the inside: keep the
    # nearest positive crossing of each of the four bounding lines.
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, (lo, hi) in enumerate(((0.0, grid_size), (0.0, grid_size))):
            d = directions[:, axis]
            safe = np.abs(d) > 1e-9
            for bound in (lo, hi):
                t = np.full(angles.shape[0], np.inf, dtype=np.float32)
                t[safe] = (bound - origin[axis]) / d[safe]
                t[t < 0.0] = np.inf
                hits = np.minimum(hits, t)

    if len(obstacles) > 0:
        rects = np.asarray(obstacles, dtype=np.float32)
        mins = rects[:, :2]
        maxs = mins + rects[:, 2:]

        # Vectorized slab test: (R, O) entry/exit parameters per ray/rectangle.
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dir = np.where(
                np.abs(directions) > 1e-9, 1.0 / directions, np.float32(np.inf)
            ).astype(np.float32)
            t_lo = (mins[None, :, :] - origin[None, None, :]) * inv_dir[:, None, :]
            t_hi = (maxs[None, :, :] - origin[None, None, :]) * inv_dir[:, None, :]

        t_near = np.max(np.minimum(t_lo, t_hi), axis=2)
        t_far = np.min(np.maximum(t_lo, t_hi), axis=2)
        valid = (t_far >= np.maximum(t_near, 0.0)) & (t_far >= 0.0)
        entry = np.where(valid, np.maximum(t_near, 0.0), np.float32(np.inf))
        hits = np.minimum(hits, np.min(entry, axis=1))

    return np.clip(np.nan_to_num(hits, nan=max_range, posinf=max_range), 0.0, max_range).astype(
        np.float32
    )


def ray_angles(n_rays: int, offset: float = 0.0) -> np.ndarray:
    """Return *n_rays* evenly spaced ray directions starting at *offset*."""
    return (
        np.linspace(0.0, 2.0 * np.pi, int(n_rays), endpoint=False, dtype=np.float32)
        + np.float32(offset)
    ).astype(np.float32)
