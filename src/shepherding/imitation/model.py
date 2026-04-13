"""Behavioral cloning model training and inference."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from shepherding.imitation.features import feature_names, observation_to_features


@dataclass
class BehavioralCloningAgent:
    """Adapter that exposes an SB3-style predict() method."""

    estimator: Any
    n_sheep: int
    max_obstacles: int
    model_feature_names: list[str]
    sentinel: float = 999.0

    def predict(
        self,
        observation: np.ndarray,
        state: object | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del state, episode_start, deterministic
        features = observation_to_features(
            observation,
            self.n_sheep,
            sentinel=self.sentinel,
        )
        model_input = _format_model_input(features, self.model_feature_names)
        action = np.asarray(self.estimator.predict(model_input)[0], dtype=np.float32)
        norm = float(np.linalg.norm(action))
        if norm > 1e-8:
            action = action / norm
        return action.astype(np.float32), None


def train_behavioral_cloning_model(
    dataset_path: Path,
    output_dir: Path,
    train_config: Dict[str, Any],
) -> dict[str, Any]:
    """Fit a multi-output random forest on demonstration data."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for behavioral cloning. Install the project "
            "dependencies after updating pyproject.toml."
        ) from exc
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for behavioral cloning training."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_path)
    n_sheep = int(train_config["n_sheep"])
    max_obstacles = int(train_config["max_obstacles"])
    features = feature_names(n_sheep, max_obstacles)
    target_cols = ["target_dx", "target_dy"]
    val_fraction = float(train_config.get("validation_fraction", 0.2))
    seed = int(train_config.get("seed", 0))

    episode_ids = np.array(sorted(df["episode_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)
    val_count = max(1, int(np.ceil(len(episode_ids) * val_fraction)))
    val_episodes = set(episode_ids[:val_count].tolist())

    train_df = df[~df["episode_id"].isin(val_episodes)].copy()
    val_df = df[df["episode_id"].isin(val_episodes)].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Need at least one training and one validation episode.")

    estimator = RandomForestRegressor(
        n_estimators=int(train_config.get("n_estimators", 300)),
        max_depth=train_config.get("max_depth"),
        min_samples_leaf=int(train_config.get("min_samples_leaf", 1)),
        random_state=seed,
        n_jobs=int(train_config.get("n_jobs", -1)),
    )
    estimator.fit(train_df[features], train_df[target_cols])

    val_pred = np.asarray(estimator.predict(val_df[features]), dtype=np.float32)
    val_true = val_df[target_cols].to_numpy(dtype=np.float32)
    angle_error = _mean_angle_error_degrees(val_true, val_pred)
    metrics = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "validation_fraction": val_fraction,
        "mse": float(mean_squared_error(val_true, val_pred)),
        "rmse": float(np.sqrt(mean_squared_error(val_true, val_pred))),
        "mae": float(mean_absolute_error(val_true, val_pred)),
        "mean_angle_error_deg": float(angle_error),
        "r2_dx": float(r2_score(val_true[:, 0], val_pred[:, 0])),
        "r2_dy": float(r2_score(val_true[:, 1], val_pred[:, 1])),
    }

    payload = {
        "estimator": estimator,
        "n_sheep": n_sheep,
        "max_obstacles": max_obstacles,
        "sentinel": float(train_config.get("sentinel", 999.0)),
        "feature_names": features,
    }
    model_path = output_dir / "behavioral_cloning.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)

    predictions = val_df[
        ["episode_id", "step", "scenario", "split", "target_dx", "target_dy"]
    ].copy()
    predictions["pred_dx"] = val_pred[:, 0]
    predictions["pred_dy"] = val_pred[:, 1]
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)

    importances = pd.DataFrame(
        {
            "feature": features,
            "importance": np.asarray(estimator.feature_importances_, dtype=np.float32),
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(output_dir / "feature_importances.csv", index=False)

    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "metadata.json",
        {
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "model_type": "behavioral_cloning",
            "train_config": json.loads(json.dumps(train_config)),
        },
    )
    return metrics


def load_behavioral_cloning_agent(path: Path | str) -> BehavioralCloningAgent:
    model_path = Path(path)
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    return BehavioralCloningAgent(
        estimator=payload["estimator"],
        n_sheep=int(payload["n_sheep"]),
        max_obstacles=int(payload.get("max_obstacles", 0)),
        model_feature_names=list(payload.get("feature_names", [])),
        sentinel=float(payload.get("sentinel", 999.0)),
    )


def _mean_angle_error_degrees(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_norm = np.linalg.norm(y_true, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(y_pred, axis=1, keepdims=True)
    true_unit = y_true / np.clip(true_norm, 1e-8, None)
    pred_unit = y_pred / np.clip(pred_norm, 1e-8, None)
    cosine = np.sum(true_unit * pred_unit, axis=1)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cosine))))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _format_model_input(features: np.ndarray, feature_names: list[str]) -> Any:
    row = np.asarray(features, dtype=np.float32).reshape(1, -1)
    if len(feature_names) != row.shape[1]:
        return row
    try:
        import pandas as pd
    except ImportError:
        return row
    return pd.DataFrame(row, columns=feature_names)
