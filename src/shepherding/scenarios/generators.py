"""Procedural scenario generation for domain randomization.

The original training distribution sampled 3-8 axis-aligned blobs of side
0.8-1.8, always placed a goal in the top-right quadrant and always spawned the
flock in the bottom-left. The held-out scenarios, by contrast, are corridors,
gates and split fields built from walls up to nine units long. A policy trained
on blobs has simply never seen a wall, which is a distribution mismatch rather
than a failure of learning.

This module samples *topologies* — open fields, blobs, corridors, gates and
bars — so that the structures used at test time are instances of a family the
agent trains on, without any specific test layout ever being generated. It also
decorrelates the goal from the spawn region so the policy cannot fall back on a
fixed "head up and to the right" prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


Rect = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass(frozen=True)
class LayoutSpec:
    """Parameter ranges for one procedural obstacle distribution."""

    topology_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "open": 0.12,
            "blobs": 0.28,
            "corridor": 0.20,
            "gate": 0.20,
            "bars": 0.20,
        }
    )
    n_blobs: Tuple[int, int] = (2, 8)
    blob_size: Tuple[float, float] = (0.8, 2.6)
    wall_thickness: Tuple[float, float] = (0.8, 1.4)
    wall_length_frac: Tuple[float, float] = (0.25, 0.55)
    gate_width: Tuple[float, float] = (3.0, 6.0)
    n_bars: Tuple[int, int] = (2, 4)
    bar_length_frac: Tuple[float, float] = (0.18, 0.40)
    n_corridor_walls: Tuple[int, int] = (2, 4)


@dataclass(frozen=True)
class DynamicsSpec:
    """Parameter ranges for randomized sheep and sensing dynamics."""

    n_sheep: Tuple[int, int] = (6, 14)
    sheep_speed: Tuple[float, float] = (0.24, 0.42)
    cohesion_factor: Tuple[float, float] = (0.04, 0.12)
    repulsion_strength: Tuple[float, float] = (0.8, 1.5)
    leader_factor: Tuple[float, float] = (0.01, 0.07)
    obstacle_avoidance_threshold: Tuple[float, float] = (1.1, 2.3)
    flee_radius: Tuple[float, float] = (4.5, 6.5)
    visibility_radius: Tuple[float, float] = (5.0, 9.5)
    observation_noise: Tuple[float, float] = (0.0, 0.12)
    action_noise: Tuple[float, float] = (0.0, 0.10)


#: Distribution used for training and for the in-distribution validation split.
TRAIN_LAYOUT_SPEC = LayoutSpec()

#: Deliberately harder, disjoint ranges for the held-out procedural test suite:
#: tighter gates, longer walls and denser blob fields than anything seen while
#: training. Reporting on this instead of the five hand-authored presets is what
#: makes the generalization number meaningful.
TEST_LAYOUT_SPEC = LayoutSpec(
    topology_weights={
        "open": 0.08,
        "blobs": 0.22,
        "corridor": 0.24,
        "gate": 0.24,
        "bars": 0.22,
    },
    n_blobs=(5, 10),
    blob_size=(1.0, 3.0),
    wall_thickness=(0.9, 1.6),
    wall_length_frac=(0.35, 0.70),
    gate_width=(2.2, 4.0),
    n_bars=(3, 5),
    bar_length_frac=(0.25, 0.50),
    n_corridor_walls=(3, 5),
)

TRAIN_DYNAMICS_SPEC = DynamicsSpec()

TEST_DYNAMICS_SPEC = DynamicsSpec(
    n_sheep=(8, 16),
    sheep_speed=(0.22, 0.46),
    cohesion_factor=(0.03, 0.13),
    repulsion_strength=(0.7, 1.6),
    leader_factor=(0.01, 0.08),
    obstacle_avoidance_threshold=(1.0, 2.5),
    flee_radius=(4.0, 7.0),
    visibility_radius=(4.5, 10.0),
    observation_noise=(0.0, 0.16),
    action_noise=(0.0, 0.14),
)


# ---------------------------------------------------------------------------
# Goal / spawn placement
# ---------------------------------------------------------------------------

def sample_goal_and_spawn(
    rng: np.random.Generator,
    grid_size: float,
    margin_frac: float = 0.12,
    min_separation_frac: float = 0.45,
    spawn_extent_frac: float = 0.14,
) -> Tuple[Point, Tuple[Point, Point]]:
    """Sample a goal anywhere in the arena and an independent flock spawn box.

    Returns ``(goal, spawn_bounds)`` where ``spawn_bounds`` is the
    ``((x_lo, y_lo), (x_hi, y_hi))`` fractional box used by the environment.
    Goal and spawn are drawn independently subject only to a minimum separation,
    which breaks the "goal is always diagonally opposite the spawn" regularity.
    """
    lo = margin_frac
    hi = 1.0 - margin_frac
    min_separation = float(min_separation_frac)

    goal = rng.uniform(lo, hi, size=2)
    for _ in range(64):
        centre = rng.uniform(lo, hi, size=2)
        if float(np.linalg.norm(centre - goal)) >= min_separation:
            break
    else:
        centre = np.clip(goal + np.array([-min_separation, -min_separation]), lo, hi)

    half = float(spawn_extent_frac) * 0.5
    x_lo = float(np.clip(centre[0] - half, 0.05, 0.95 - 2 * half))
    y_lo = float(np.clip(centre[1] - half, 0.05, 0.95 - 2 * half))
    spawn_bounds = ((x_lo, y_lo), (x_lo + 2 * half, y_lo + 2 * half))
    goal_point = (float(goal[0] * grid_size), float(goal[1] * grid_size))
    return goal_point, spawn_bounds


# ---------------------------------------------------------------------------
# Obstacle topologies
# ---------------------------------------------------------------------------

def sample_layout(
    rng: np.random.Generator,
    grid_size: float,
    spec: LayoutSpec,
    keepouts: Sequence[Point],
    keepout_radius: float,
    max_obstacles: int,
    topology: Optional[str] = None,
) -> Tuple[str, List[Rect]]:
    """Sample an obstacle layout, retrying until start and goal stay connected.

    Returns ``(topology_name, obstacles)``.
    """
    if topology is None:
        names = list(spec.topology_weights.keys())
        weights = np.asarray([spec.topology_weights[name] for name in names], dtype=np.float64)
        weights = weights / weights.sum()
        topology = str(rng.choice(names, p=weights))

    builders = {
        "open": _build_open,
        "blobs": _build_blobs,
        "corridor": _build_corridor,
        "gate": _build_gate,
        "bars": _build_bars,
    }
    builder = builders.get(topology)
    if builder is None:
        raise ValueError(f"Unknown topology '{topology}'.")

    for _ in range(12):
        obstacles = builder(rng, grid_size, spec)
        obstacles = [
            rect
            for rect in obstacles
            if not _rect_near_any(rect, keepouts, keepout_radius)
        ][:max_obstacles]
        if _is_connected(obstacles, grid_size, keepouts):
            return topology, obstacles

    # Every attempt sealed the arena; fall back to an empty field rather than
    # handing the agent an unsolvable episode.
    return "open", []


def _build_open(
    rng: np.random.Generator, grid_size: float, spec: LayoutSpec
) -> List[Rect]:
    return []


def _build_blobs(
    rng: np.random.Generator, grid_size: float, spec: LayoutSpec
) -> List[Rect]:
    count = int(rng.integers(spec.n_blobs[0], spec.n_blobs[1] + 1))
    obstacles: List[Rect] = []
    for _ in range(count):
        for _ in range(32):
            width = float(rng.uniform(*spec.blob_size))
            height = float(rng.uniform(*spec.blob_size))
            centre = rng.uniform(grid_size * 0.12, grid_size * 0.88, size=2)
            rect = _rect_from_centre(centre, width, height, grid_size)
            if not _overlaps_any(rect, obstacles, padding=0.5):
                obstacles.append(rect)
                break
    return obstacles


def _build_corridor(
    rng: np.random.Generator, grid_size: float, spec: LayoutSpec
) -> List[Rect]:
    """Parallel long walls forming a channel, in either orientation."""
    vertical = bool(rng.random() < 0.5)
    count = int(rng.integers(spec.n_corridor_walls[0], spec.n_corridor_walls[1] + 1))
    obstacles: List[Rect] = []
    positions = np.sort(rng.uniform(0.22, 0.78, size=count)) * grid_size
    for position in positions:
        thickness = float(rng.uniform(*spec.wall_thickness))
        length = float(rng.uniform(*spec.wall_length_frac)) * grid_size
        start = float(rng.uniform(0.0, grid_size - length))
        if vertical:
            rect = (float(position), start, thickness, length)
        else:
            rect = (start, float(position), length, thickness)
        obstacles.append(_clamp_rect(rect, grid_size))
    return obstacles


def _build_gate(
    rng: np.random.Generator, grid_size: float, spec: LayoutSpec
) -> List[Rect]:
    """A wall spanning the arena with a single opening of random width/offset."""
    vertical = bool(rng.random() < 0.5)
    position = float(rng.uniform(0.32, 0.68)) * grid_size
    thickness = float(rng.uniform(*spec.wall_thickness))
    gate_width = float(rng.uniform(*spec.gate_width))
    gate_centre = float(rng.uniform(gate_width, grid_size - gate_width))

    lower_len = max(gate_centre - gate_width / 2.0, 0.0)
    upper_start = min(gate_centre + gate_width / 2.0, grid_size)
    upper_len = max(grid_size - upper_start, 0.0)

    obstacles: List[Rect] = []
    if lower_len > 0.3:
        rect = (position, 0.0, thickness, lower_len) if vertical else (0.0, position, lower_len, thickness)
        obstacles.append(_clamp_rect(rect, grid_size))
    if upper_len > 0.3:
        rect = (
            (position, upper_start, thickness, upper_len)
            if vertical
            else (upper_start, position, upper_len, thickness)
        )
        obstacles.append(_clamp_rect(rect, grid_size))
    return obstacles


def _build_bars(
    rng: np.random.Generator, grid_size: float, spec: LayoutSpec
) -> List[Rect]:
    """Offset long thin bars, the family that contains the split-field preset."""
    vertical = bool(rng.random() < 0.5)
    count = int(rng.integers(spec.n_bars[0], spec.n_bars[1] + 1))
    obstacles: List[Rect] = []
    for _ in range(count):
        thickness = float(rng.uniform(*spec.wall_thickness))
        length = float(rng.uniform(*spec.bar_length_frac)) * grid_size
        along = float(rng.uniform(0.08, 0.92)) * grid_size
        across = float(rng.uniform(0.12, 0.88)) * grid_size
        rect = (
            (across, along, thickness, length) if vertical else (along, across, length, thickness)
        )
        candidate = _clamp_rect(rect, grid_size)
        if not _overlaps_any(candidate, obstacles, padding=0.4):
            obstacles.append(candidate)
    return obstacles


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rect_from_centre(
    centre: np.ndarray, width: float, height: float, grid_size: float
) -> Rect:
    x = float(np.clip(centre[0] - width / 2.0, 0.4, grid_size - width - 0.4))
    y = float(np.clip(centre[1] - height / 2.0, 0.4, grid_size - height - 0.4))
    return (x, y, width, height)


def _clamp_rect(rect: Rect, grid_size: float) -> Rect:
    x, y, w, h = rect
    w = float(min(w, grid_size))
    h = float(min(h, grid_size))
    x = float(np.clip(x, 0.0, grid_size - w))
    y = float(np.clip(y, 0.0, grid_size - h))
    return (x, y, w, h)


def _overlaps_any(rect: Rect, existing: Sequence[Rect], padding: float) -> bool:
    rx, ry, rw, rh = rect
    for ex, ey, ew, eh in existing:
        separated = (
            rx + rw + padding < ex
            or ex + ew + padding < rx
            or ry + rh + padding < ey
            or ey + eh + padding < ry
        )
        if not separated:
            return True
    return False


def _rect_near_any(rect: Rect, points: Sequence[Point], radius: float) -> bool:
    """True if *rect* comes within *radius* of any keep-out point."""
    rx, ry, rw, rh = rect
    for px, py in points:
        cx = float(np.clip(px, rx, rx + rw))
        cy = float(np.clip(py, ry, ry + rh))
        if float(np.hypot(px - cx, py - cy)) < radius:
            return True
    return False


def _is_connected(
    obstacles: Sequence[Rect],
    grid_size: float,
    points: Sequence[Point],
    resolution: int = 40,
) -> bool:
    """Coarse flood fill checking that all keep-out points share a free region.

    Without this, a randomly sampled gate or corridor can seal the goal off and
    produce an episode the agent cannot solve at any skill level, which shows up
    as unexplained variance in the success rate.
    """
    if len(points) < 2:
        return True

    cell = grid_size / float(resolution)
    blocked = np.zeros((resolution, resolution), dtype=bool)
    for rx, ry, rw, rh in obstacles:
        i0 = max(int(np.floor(rx / cell)), 0)
        i1 = min(int(np.ceil((rx + rw) / cell)), resolution)
        j0 = max(int(np.floor(ry / cell)), 0)
        j1 = min(int(np.ceil((ry + rh) / cell)), resolution)
        blocked[i0:i1, j0:j1] = True

    def to_cell(point: Point) -> Tuple[int, int]:
        return (
            int(np.clip(int(point[0] / cell), 0, resolution - 1)),
            int(np.clip(int(point[1] / cell), 0, resolution - 1)),
        )

    start = to_cell(points[0])
    targets = {to_cell(point) for point in points[1:]}
    if blocked[start] or any(blocked[target] for target in targets):
        return False

    seen = np.zeros_like(blocked)
    seen[start] = True
    stack = [start]
    while stack:
        i, j = stack.pop()
        targets.discard((i, j))
        if not targets:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < resolution and 0 <= nj < resolution:
                if not seen[ni, nj] and not blocked[ni, nj]:
                    seen[ni, nj] = True
                    stack.append((ni, nj))
    return not targets
