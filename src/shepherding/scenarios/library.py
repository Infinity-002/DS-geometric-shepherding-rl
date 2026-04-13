"""Named scenario families for the research shepherding environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


Rect = Tuple[float, float, float, float]
SpawnBounds = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class ScenarioTemplate:
    goal: Tuple[float, float]
    visibility_radius: float
    obstacles: Tuple[Rect, ...]
    sheep_speed: float
    cohesion_factor: float
    repulsion_strength: float
    leader_factor: float
    obstacle_avoidance_threshold: float
    spawn_bounds: Optional[SpawnBounds] = None


def default_spawn_bounds() -> SpawnBounds:
    return ((0.15, 0.15), (0.45, 0.45))


def fixed_training_obstacles(grid_size: float) -> list[Rect]:
    return [
        (grid_size * 0.22, grid_size * 0.28, 1.2, 1.0),
        (grid_size * 0.38, grid_size * 0.54, 1.4, 0.9),
        (grid_size * 0.56, grid_size * 0.30, 1.0, 1.3),
        (grid_size * 0.68, grid_size * 0.60, 1.1, 1.1),
    ]


def structured_training_obstacle_layouts(grid_size: float) -> list[tuple[Rect, ...]]:
    """Preset obstacle layouts with fixed shapes but shifted placements."""
    return [
        (
            (grid_size * 0.22, grid_size * 0.28, 1.2, 1.0),
            (grid_size * 0.38, grid_size * 0.54, 1.4, 0.9),
            (grid_size * 0.56, grid_size * 0.30, 1.0, 1.3),
            (grid_size * 0.68, grid_size * 0.60, 1.1, 1.1),
        ),
        (
            (grid_size * 0.18, grid_size * 0.34, 1.2, 1.0),
            (grid_size * 0.34, grid_size * 0.60, 1.4, 0.9),
            (grid_size * 0.58, grid_size * 0.24, 1.0, 1.3),
            (grid_size * 0.72, grid_size * 0.50, 1.1, 1.1),
        ),
        (
            (grid_size * 0.26, grid_size * 0.22, 1.2, 1.0),
            (grid_size * 0.42, grid_size * 0.48, 1.4, 0.9),
            (grid_size * 0.60, grid_size * 0.36, 1.0, 1.3),
            (grid_size * 0.66, grid_size * 0.66, 1.1, 1.1),
        ),
        (
            (grid_size * 0.20, grid_size * 0.24, 1.2, 1.0),
            (grid_size * 0.46, grid_size * 0.58, 1.4, 0.9),
            (grid_size * 0.52, grid_size * 0.40, 1.0, 1.3),
            (grid_size * 0.70, grid_size * 0.68, 1.1, 1.1),
        ),
    ]


def opposite_goal_spawn_bounds(
    goal: tuple[float, float],
    grid_size: float,
    low_band: tuple[float, float] = (0.12, 0.18),
    high_band: tuple[float, float] = (0.36, 0.42),
) -> SpawnBounds:
    """Spawn flock in the corner opposite the goal."""
    mid = grid_size * 0.5
    goal_x, goal_y = goal

    if goal_x >= mid:
        x_bounds = (low_band[0], high_band[0])
    else:
        x_bounds = (1.0 - high_band[0], 1.0 - low_band[0])

    if goal_y >= mid:
        y_bounds = (low_band[1], high_band[1])
    else:
        y_bounds = (1.0 - high_band[1], 1.0 - low_band[1])

    return ((x_bounds[0], y_bounds[0]), (x_bounds[1], y_bounds[1]))


def scenario_presets(grid_size: float) -> Dict[str, ScenarioTemplate]:
    return {
        "unseen_corridor": ScenarioTemplate(
            goal=(grid_size * 0.86, grid_size * 0.85),
            visibility_radius=6.2,
            obstacles=(
                (grid_size * 0.34, grid_size * 0.18, 1.1, 6.8),
                (grid_size * 0.34, grid_size * 0.64, 1.1, 4.4),
                (grid_size * 0.56, grid_size * 0.10, 1.1, 5.2),
                (grid_size * 0.56, grid_size * 0.48, 1.1, 7.0),
            ),
            sheep_speed=0.31,
            cohesion_factor=0.08,
            repulsion_strength=1.15,
            leader_factor=0.05,
            obstacle_avoidance_threshold=1.9,
        ),
        "unseen_dense": ScenarioTemplate(
            goal=(grid_size * 0.84, grid_size * 0.88),
            visibility_radius=5.8,
            obstacles=(
                (grid_size * 0.26, grid_size * 0.36, 1.2, 1.6),
                (grid_size * 0.42, grid_size * 0.24, 1.5, 1.1),
                (grid_size * 0.52, grid_size * 0.54, 1.3, 1.7),
                (grid_size * 0.64, grid_size * 0.34, 1.4, 1.2),
                (grid_size * 0.42, grid_size * 0.72, 1.6, 1.0),
            ),
            sheep_speed=0.35,
            cohesion_factor=0.06,
            repulsion_strength=1.25,
            leader_factor=0.03,
            obstacle_avoidance_threshold=1.7,
            spawn_bounds=((0.12, 0.18), (0.34, 0.42)),
        ),
        "unseen_narrow_gate": ScenarioTemplate(
            goal=(grid_size * 0.85, grid_size * 0.82),
            visibility_radius=6.4,
            obstacles=(
                (grid_size * 0.40, grid_size * 0.08, 1.1, 6.6),
                (grid_size * 0.40, grid_size * 0.76, 1.1, 3.2),
                (grid_size * 0.58, grid_size * 0.00, 1.1, 2.8),
                (grid_size * 0.58, grid_size * 0.40, 1.1, 9.0),
            ),
            sheep_speed=0.32,
            cohesion_factor=0.08,
            repulsion_strength=1.10,
            leader_factor=0.05,
            obstacle_avoidance_threshold=2.0,
            spawn_bounds=((0.12, 0.22), (0.30, 0.40)),
        ),
        "unseen_split_field": ScenarioTemplate(
            goal=(grid_size * 0.80, grid_size * 0.86),
            visibility_radius=6.9,
            obstacles=(
                (grid_size * 0.28, grid_size * 0.46, 2.4, 0.9),
                (grid_size * 0.56, grid_size * 0.28, 2.6, 0.9),
                (grid_size * 0.56, grid_size * 0.64, 2.6, 0.9),
            ),
            sheep_speed=0.34,
            cohesion_factor=0.07,
            repulsion_strength=1.20,
            leader_factor=0.04,
            obstacle_avoidance_threshold=1.8,
            spawn_bounds=((0.15, 0.12), (0.36, 0.32)),
        ),
        "unseen_open_field": ScenarioTemplate(
            goal=(grid_size * 0.88, grid_size * 0.84),
            visibility_radius=5.6,
            obstacles=(),
            sheep_speed=0.36,
            cohesion_factor=0.06,
            repulsion_strength=1.05,
            leader_factor=0.03,
            obstacle_avoidance_threshold=1.5,
            spawn_bounds=((0.18, 0.18), (0.42, 0.42)),
        ),
    }


def available_scenarios(grid_size: float) -> list[str]:
    return sorted(scenario_presets(grid_size).keys())
