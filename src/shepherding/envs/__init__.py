"""Shepherding environments package."""

import gymnasium as gym

gym.register(
    id="HerdingEnv-v0",
    entry_point="shepherding.envs.herding_env:HerdingEnv",
)
