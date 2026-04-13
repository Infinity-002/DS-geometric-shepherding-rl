"""Shepherding environments package."""

import gymnasium as gym

# ── v1: original prototype (unchanged) ──────────────────────────────────────
gym.register(
    id="HerdingEnv-v0",
    entry_point="shepherding.envs.herding_env:HerdingEnv",
)

# ── v2: limited visibility + obstacles + autonomous sheep ───────────────────
gym.register(
    id="HerdingEnv-v2",
    entry_point="shepherding.envs.herding_env_v2:HerdingEnvV2",
)

# ── v3: research environment with domain randomization ──────────────────────
gym.register(
    id="HerdingEnv-v3",
    entry_point="shepherding.envs.herding_env_v3:HerdingEnvV3",
)
