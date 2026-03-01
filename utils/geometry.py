"""
Geometric utilities for flock analysis and reward computation.

Provides convex-hull computation, centroid calculation, and a composite
reward function that captures centroid progress, perimeter penalty, and
incursion (blocking) penalty.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Convex Hull helpers
# ---------------------------------------------------------------------------

def compute_convex_hull(positions: np.ndarray) -> Optional[ConvexHull]:
    """Compute the 2-D convex hull of *positions*.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape ``(N, 2)`` with the 2-D coordinates.

    Returns
    -------
    ConvexHull or None
        The convex hull, or ``None`` when the point set is degenerate
        (fewer than 3 unique points or all collinear).
    """
    if positions.shape[0] < 3:
        return None
    try:
        return ConvexHull(positions)
    except Exception:
        # QHull errors on degenerate inputs (collinear, duplicate, etc.)
        return None


def compute_centroid(positions: np.ndarray) -> np.ndarray:
    """Return the mean position (centroid) of *positions*.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape ``(N, 2)``.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` centroid.
    """
    return np.mean(positions, axis=0)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(
    sheep_positions: np.ndarray,
    dog_position: np.ndarray,
    goal: np.ndarray,
    *,
    grid_size: float = 20.0,
    w_centroid: float = 1.0,
    w_perimeter: float = 0.1,
    w_incursion: float = 0.5,
    w_proximity: float = 0.3,
) -> float:
    """Compute the composite geometric reward.

    The reward is a weighted sum of four terms, all normalised to roughly
    the ``[-1, 0]`` range so learning is stable:

    1. **Centroid progress** – negative normalised distance from the flock
       centroid to the *goal*.
    2. **Perimeter penalty** – negative convex-hull area (normalised by
       grid area) to encourage tight flocking.
    3. **Incursion (blocking) penalty** – penalises the dog for being
       closer to the goal than the flock centroid.
    4. **Dog-to-flock proximity** – rewards the dog for being close to
       the flock, encouraging it to approach and herd.

    Parameters
    ----------
    sheep_positions : np.ndarray
        Shape ``(N, 2)`` sheep coordinates.
    dog_position : np.ndarray
        Shape ``(2,)`` dog coordinate.
    goal : np.ndarray
        Shape ``(2,)`` goal coordinate.
    grid_size : float
        Size of the grid, used for normalisation.
    w_centroid : float
        Weight for the centroid-distance term.
    w_perimeter : float
        Weight for the perimeter / area penalty.
    w_incursion : float
        Weight for the blocking penalty.
    w_proximity : float
        Weight for the dog-to-flock proximity bonus.

    Returns
    -------
    float
        Scalar reward value.
    """
    centroid = compute_centroid(sheep_positions)
    max_dist: float = float(np.sqrt(2.0) * grid_size)  # diagonal

    # --- 1. Centroid progress (normalised negative distance to goal) -------
    centroid_dist: float = float(np.linalg.norm(centroid - goal))
    centroid_reward: float = -(centroid_dist / max_dist)

    # --- 2. Perimeter penalty (normalised negative hull area) -------------
    hull = compute_convex_hull(sheep_positions)
    grid_area: float = grid_size * grid_size
    if hull is not None:
        hull_metric: float = float(hull.volume)  # 2-D → area
    else:
        # Fallback: bounding-box area
        mins = sheep_positions.min(axis=0)
        maxs = sheep_positions.max(axis=0)
        hull_metric = float(np.prod(maxs - mins))
    perimeter_penalty: float = -(hull_metric / grid_area)

    # --- 3. Incursion (blocking) penalty ----------------------------------
    dog_goal_dist: float = float(np.linalg.norm(dog_position - goal))
    incursion_penalty: float = -1.0 if dog_goal_dist < centroid_dist else 0.0

    # --- 4. Dog-to-flock proximity bonus ----------------------------------
    dog_flock_dist: float = float(np.linalg.norm(dog_position - centroid))
    proximity_bonus: float = -(dog_flock_dist / max_dist)

    # --- Composite reward -------------------------------------------------
    reward: float = (
        w_centroid * centroid_reward
        + w_perimeter * perimeter_penalty
        + w_incursion * incursion_penalty
        + w_proximity * proximity_bonus
    )
    return reward
