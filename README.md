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
uv run python scripts/train_v3_recurrent.py --config configs/research/v3.yaml --seed 0
```

This trains recurrent PPO with the adaptive curriculum, keeps the checkpoint that
scores best on the `validation` split, and saves its VecNormalize statistics next to
it (`recurrent_seed0_best_vecnormalize.pkl`). `configs/research/v3_gate045.yaml` is
the curriculum ablation reported in the paper.

### Evaluate on the Held-Out Suite

```bash
uv run python scripts/evaluate_generalization.py \
  --model-type recurrent \
  --model-path models/research_v3/recurrent/recurrent_seed0_best.zip \
  --vecnormalize models/research_v3/recurrent/recurrent_seed0_best_vecnormalize.pkl \
  --episodes 150

# Baselines on the same episodes
uv run python scripts/evaluate_generalization.py --model-type heuristic --episodes 150
```

**Always pass `--vecnormalize`.** Training normalizes observations, so a model
evaluated on raw observations scores near-randomly. Pass `--fixed-sheep-count` to
compare PPO with the heuristic and behavioral cloning on identical 10-sheep episodes.

| Scenario | Role |
| --- | --- |
| `train` | Randomized training distribution, narrowed by the curriculum |
| `validation` | Full training ranges, separate seeds; used only for checkpoint selection |
| `test_procedural` | Wider ranges than training, unseen seeds; the number to report |
| `unseen_*` | Five fixed hand-designed maps, for case studies and rendering |

Report the procedural test rather than the five `unseen_*` maps: single maps vary
by up to 39 points between training seeds. `models/research_v3/<type>/<run>_metadata.json`
records `curriculum.fraction_per_stage`; a run that spent most of its steps below
stage 0.66 trained on narrower randomization than the config asks for.

The full record of training runs, evaluations and fixes is in
[`docs/experiments/run_log.md`](docs/experiments/run_log.md), and the paper sources
are in [`docs/paper/`](docs/paper/).

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
