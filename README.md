# Geometric-Informed Reinforcement Learning for the Shepherding Problem

A PPO-based reinforcement learning agent that solves the **shepherding problem** — training a single dog to herd a flock of heuristic sheep toward a goal on a continuous 2D grid.

Sheep follow the **Strömbom et al. (2014)** behavioural model with flee, cohesion, and repulsion forces. The reward function uses **geometric signals** (centroid progress, convex-hull compactness, incursion penalty, and proximity bonus).

## Project Structure

```
├── src/shepherding/
│   ├── baselines/          # Heuristic controllers and non-RL baselines
│   ├── envs/               # Gymnasium environments
│   ├── research/           # Training, evaluation, benchmark, curriculum utilities
│   ├── scenarios/          # Deterministic scenario library for v3
│   └── utils/              # Shared geometry helpers
├── scripts/                # Thin training/evaluation/benchmark entry points
├── configs/
│   └── research/           # v3 training and benchmark configs
├── tests/                  # Regression tests for env dynamics and research utilities
├── models/                 # Saved models (gitignored)
├── docs/                   # Documentation notebooks
├── pyproject.toml          # Project metadata & dependencies
└── README.md
```

## Documentation

- **[Stage 1 Documentation](docs/stage1_documentation.ipynb)**: Complete walkthrough of the basic herding setup, including problem formulation, implementation details, training process, evaluation results, and visualizations.

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

## Presentation Dashboard

For a presentation-only Streamlit app that uses the exported PNGs and GIFs
from `images/` directly:

```bash
uv add streamlit
uv run streamlit run streamlit_app.py
```

The app does not launch training or evaluation code. It only displays the
assets already present in `images/`.

## Research Pipeline (v3)

The existing `v0` and `v2` pipelines remain unchanged. A separate `v3`
research track adds:

- recurrent PPO for partial observability
- domain-randomized training environments
- unseen scenario evaluation
- multi-seed baselines and ablations
- trajectory export for downstream analysis
- publication-style figures and summary tables

### Train v3 models

```bash
uv run python scripts/train_v3_feedforward.py --seed 0
uv run python scripts/train_v3_recurrent.py --seed 0
```

### Benchmark v3 models

```bash
uv run python scripts/benchmark_v3.py
```

### Run ablations

```bash
uv run python scripts/run_ablations_v3.py --seeds 0 1 2 --save-models
```

### Analyze outputs

```bash
uv run python scripts/analyze_results_v3.py
```

The starter notebook for exploratory analysis lives at
`docs/research_analysis_v3.ipynb`.

## Data Science Extension: Behavioral Cloning

Alongside the PPO agents, the repo now includes a supervised-learning
pipeline for a clean data-science comparison:

- a cluster-aware heuristic expert that produces demonstrations
- behavioral cloning with a random-forest regressor
- offline metrics such as MSE, RMSE, MAE, angle error, and feature importance
- online benchmark comparison against heuristic and RL agents

### Generate demonstrations

```bash
uv run python scripts/generate_bc_dataset.py
```

This exports demonstration rows to `results/imitation/dataset/` with:

- raw observations
- engineered geometric features
- expert target actions `(dx, dy)`

### Train the behavioral cloning model

```bash
uv run python scripts/train_bc.py
```

The trained model and offline validation metrics are saved under
`models/imitation/random_forest/`.

### Benchmark heuristic vs BC vs RL

```bash
uv run python scripts/benchmark_v3.py \
  --benchmark-config configs/research/benchmark_v3_ds.yaml \
  --output-dir results/research_v3/ds_benchmark
```

This comparison uses the same environment metrics as RL evaluation:

- success rate
- episode return
- episode length
- mean distance to goal
- flock spread / hull area
- stray count
- collision count

### Fast comparison path

If full v3 training takes too long, use the reduced experiment setup:

```bash
uv run python scripts/generate_bc_dataset.py --config configs/research/v3_fast.yaml --output-dir results/imitation_fast/dataset
uv run python scripts/train_bc.py --config configs/research/v3_fast.yaml --dataset-path results/imitation_fast/dataset/demonstrations.csv --output-dir models/imitation_fast/random_forest
uv run python scripts/train_v3_recurrent.py --config configs/research/v3_fast.yaml --seed 0 --run-name recurrent_fast_seed0
uv run python scripts/benchmark_v3.py --config configs/research/v3_fast.yaml --benchmark-config configs/research/benchmark_v3_fast_ds.yaml --output-dir results/research_v3_fast/ds_benchmark
```

This fast track uses:

- one seed instead of multiple seeds
- `120000` RL timesteps instead of the full research budget
- only `train`, `unseen_split_field`, and `unseen_open_field`
- fewer evaluation episodes

It is the recommended setup when you want a report-ready comparison
between heuristic, behavioral cloning, and RL without the full training cost.

### Stronger RL training path

If the fast RL checkpoint is still not solving enough episodes, use the
stronger recurrent preset:

```bash
uv run python scripts/train_v3_recurrent.py --config configs/research/v3_improved_rl.yaml --seed 0 --run-name recurrent_improved_seed0
```

This preset increases the RL budget and restores the stronger recurrent
settings while still staying much cheaper than the full `v3.yaml` run:

- `400000` timesteps instead of `120000`
- `1024` recurrent rollout steps instead of `512`
- `256` LSTM hidden size instead of `128`
- `650` max episode steps for better endgame completion

### Curriculum Learning (v3 Structured)

For even more robust training, especially in obstacle-dense environments, you can use the **Adaptive Curriculum** pipeline. This advances the environment difficulty (domain randomization and obstacle complexity) based on rolling metrics like success rate and visibility.

```bash
uv run python scripts/train_v3_recurrent.py --config configs/research/v3_structured.yaml --seed 0
```

The structured config also enables:
- **Structured Obstacles**: Uses fixed preset layouts instead of purely random placements for more consistent training signals.
- **Strategic Spawning**: Spawns the flock and dog in positions opposite to the goal to maximize the required herding distance.

After training, you can render the improved RL checkpoint with:

```bash
uv run python scripts/render_v3_3d.py \
  --config configs/research/v3_improved_rl.yaml \
  --model-type recurrent \
  --model-path models/research_v3_improved/recurrent/recurrent_improved_seed0.zip \
  --scenario train \
  --seed 0
```

To generate report-friendly figures and a **Data Science Dashboard** from the benchmark outputs:

```bash
uv run python scripts/analyze_ds_results.py \
  --results-dir results/research_v3_fast/ds_benchmark \
  --bc-metrics models/imitation_fast/random_forest/metrics.json
```

This creates a comprehensive suite of visualizations in `results/research_v3_fast/ds_benchmark/figures_ds/`:

- **Main Dashboard**: A unified view of success rates, returns, and efficiency.
- **Generalization Gap**: Comparison between training performance and unseen scenarios.
- **Metric Heatmap**: Performance cross-comparison across all evaluation metrics.
- **Success Scatter Plots**: Success rate vs. episode length and path efficiency.

### Run tests

```bash
uv run python -m unittest discover -s tests -t .
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
