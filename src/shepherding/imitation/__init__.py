"""Imitation-learning utilities for data-science experiments."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BehavioralCloningAgent",
    "collect_demonstrations",
    "feature_names",
    "load_behavioral_cloning_agent",
    "observation_to_features",
    "train_behavioral_cloning_model",
]


def __getattr__(name: str) -> Any:
    if name == "collect_demonstrations":
        from shepherding.imitation.dataset import collect_demonstrations

        return collect_demonstrations
    if name in {"BehavioralCloningAgent", "load_behavioral_cloning_agent", "train_behavioral_cloning_model"}:
        from shepherding.imitation.model import (
            BehavioralCloningAgent,
            load_behavioral_cloning_agent,
            train_behavioral_cloning_model,
        )

        mapping = {
            "BehavioralCloningAgent": BehavioralCloningAgent,
            "load_behavioral_cloning_agent": load_behavioral_cloning_agent,
            "train_behavioral_cloning_model": train_behavioral_cloning_model,
        }
        return mapping[name]
    if name in {"feature_names", "observation_to_features"}:
        from shepherding.imitation.features import feature_names, observation_to_features

        mapping = {
            "feature_names": feature_names,
            "observation_to_features": observation_to_features,
        }
        return mapping[name]
    raise AttributeError(name)
