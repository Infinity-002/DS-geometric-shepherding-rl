# Geometric Shepherding RL

This project studies the **shepherding problem**: how a single dog can guide a flock of sheep to a goal in a continuous 2D environment with obstacles and limited visibility.

The main goal of the project is to compare three approaches under the same environment:

- a **geometric heuristic** baseline
- **behavioral cloning** from expert demonstrations
- **reinforcement learning** with PPO / recurrent PPO

## Project Idea

The dog is the only controlled agent.  
The sheep follow rule-based flocking and escape behavior.  
The challenge is to move the flock to the goal while keeping it visible, compact, and away from obstacles.

This repository is built around a shared research environment so the different methods can be compared fairly on:

- success rate
- episode return
- distance to goal
- efficiency
- generalization to unseen scenarios

## What Is Implemented

### Heuristic Baseline

A hand-designed collect-and-drive controller based on flock geometry.

### Behavioral Cloning

A supervised learning pipeline that trains a random forest to imitate the heuristic expert from engineered geometric features.

### Reinforcement Learning

Feedforward and recurrent PPO agents trained in the same shepherding environment.  
The recurrent policy is especially useful because the environment is partially observable.

## Environment

The main environment is [`src/shepherding/envs/herding_env_v3.py`](src/shepherding/envs/herding_env_v3.py).

It includes:

- partial observability
- obstacle-aware movement
- reward shaping based on flock geometry
- domain-randomized and structured training setups
- deterministic unseen evaluation scenarios

The reward combines signals such as:

- progress toward the goal
- flock compactness
- visibility maintenance
- collision penalties
- driving the flock from a useful position

## Repository Structure

```text
src/shepherding/
├── baselines/      # Heuristic controller
├── envs/           # Environment implementations
├── imitation/      # Behavioral cloning pipeline
├── research/       # RL model building, evaluation, benchmarking
├── scenarios/      # Scenario definitions
└── utils/          # Geometry and helper functions

scripts/            # Training, evaluation, plotting, rendering
configs/            # Experiment configs
tests/              # Unit and regression tests
streamlit_app.py    # Presentation dashboard
```

## Installation

```bash
git clone https://github.com/<your-username>/geometric-shepherding-rl.git
cd geometric-shepherding-rl
uv sync
```

## Main Workflows

### Train RL

```bash
uv run python scripts/train_v3_recurrent.py --seed 0
```

### Train Behavioral Cloning

```bash
uv run python scripts/generate_bc_dataset.py
uv run python scripts/train_bc.py
```

### Run Benchmark Comparison

```bash
uv run python scripts/benchmark_v3.py \
  --config configs/research/v3_fast.yaml \
  --benchmark-config configs/research/benchmark_v3_fast_ds.yaml \
  --output-dir results/research_v3_fast/ds_benchmark
```

### Generate Comparison Figures

```bash
uv run python scripts/analyze_ds_results.py \
  --results-dir results/research_v3_fast/ds_benchmark \
  --bc-metrics models/imitation_fast/random_forest/metrics.json
```

### Launch Presentation Dashboard

```bash
uv run streamlit run streamlit_app.py
```

## Results

The project supports both:

- **offline evaluation** for behavioral cloning, such as regression error and angle error
- **online evaluation** in the environment, such as success rate and goal proximity

This makes it possible to compare not only how well a model imitates expert actions, but also how well it actually controls the flock when rolled out in the environment.

## Where To Start Reading

If you want the quickest understanding of the project, start with:

- [`src/shepherding/envs/herding_env_v3.py`](src/shepherding/envs/herding_env_v3.py)
- [`src/shepherding/baselines/heuristic.py`](src/shepherding/baselines/heuristic.py)
- [`src/shepherding/imitation/model.py`](src/shepherding/imitation/model.py)
- [`src/shepherding/research/benchmark.py`](src/shepherding/research/benchmark.py)

## Testing

```bash
uv run python -m unittest discover -s tests -t .
```

## License

MIT License. See [LICENSE](LICENSE).
