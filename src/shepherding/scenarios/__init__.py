"""Scenario definitions for the research environments."""

from shepherding.scenarios.library import (
    ScenarioTemplate,
    available_scenarios,
    default_spawn_bounds,
    fixed_training_obstacles,
    opposite_goal_spawn_bounds,
    scenario_presets,
    structured_training_obstacle_layouts,
)

__all__ = [
    "ScenarioTemplate",
    "available_scenarios",
    "default_spawn_bounds",
    "fixed_training_obstacles",
    "opposite_goal_spawn_bounds",
    "scenario_presets",
    "structured_training_obstacle_layouts",
]
