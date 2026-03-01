# Geometric-Informed Reinforcement Learning for the Shepherding Problem

A PPO-based reinforcement learning agent that solves the **shepherding problem** — training a single dog to herd a flock of heuristic sheep toward a goal on a continuous 2D grid.

Sheep follow the **Strömbom et al. (2014)** behavioural model with flee, cohesion, and repulsion forces. The reward function uses **geometric signals** (centroid progress, convex-hull compactness, incursion penalty, and proximity bonus).

## Project Structure

```
├── src/shepherding/        # Installable Python package
│   ├── envs/               # Gymnasium environment (HerdingEnv)
│   └── utils/              # Geometry helpers & reward function
├── scripts/                # Training & evaluation entry points
├── configs/                # Hyperparameter YAML files
├── models/                 # Saved models (gitignored)
├── pyproject.toml          # Project metadata & dependencies
└── README.md
```

## Installation

```bash
# Clone and install
git clone https://github.com/<your-username>/geometric-shepherding-rl.git
cd geometric-shepherding-rl
uv sync
```

## Usage

### Train

```bash
uv run python scripts/train.py
uv run python scripts/train.py --total-timesteps 1e6 --seed 42
```

### Evaluate

```bash
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --save animation.gif
```

## Key Components

| Component | Description |
|---|---|
| **HerdingEnv** | Gymnasium env with Strömbom-style sheep physics |
| **Reward function** | 4-term composite: centroid progress, perimeter penalty, incursion penalty, proximity bonus |
| **PPO Agent** | Stable-Baselines3 PPO with MLP actor-critic |

## Configuration

All hyperparameters are documented in [`configs/default.yaml`](configs/default.yaml).

## License

MIT License. See [LICENSE](LICENSE).
