# v3 experiment log

Chronological record of every v3 training run and evaluation: what was run, what
happened, and what changed because of it. Kept for writing the paper, so each
entry states the config, the code state and the numbers rather than a summary.

Conventions
- **Train metrics** are rolling means over recent training episodes (curriculum
  distribution at the current stage). **Validation** is the `validation` split:
  full breadth (1.0), randomized goal, its own seed stream, 12 episodes per
  checkpoint (30 from run 5 on).
  **Test** is `test_procedural`: wider parameter ranges that extend past the
  training ranges at both ends (supersets, not disjoint intervals), seeds from
  100000, 150 episodes. That is the number to report.
- `FracGoal` = fraction of sheep inside the goal radius at episode end (graded).
  `SR` = all sheep inside (strict). `CollEvt` = dog collision *events* per episode
  (obstacles **and arena walls**; consecutive contact counts once).
- Raw logs live in `logs/runs/`.

---

## Run 1 — 433k trial (stopped), 2026-09-10

| | |
|---|---|
| Command | `uv run python scripts/train_v3_recurrent.py --config configs/research/v3.yaml --seed 0` |
| Code | `d3730e2` (before curriculum fixes) |
| Budget | 1M planned, stopped at ~433k |
| Throughput | ~147 fps (12 subproc envs, CPU) |

**Outcome: failed.** Validation FracGoal was exactly 0.000 at all 17 checkpoints
(25k–425k); `_best.zip` was just the 25k checkpoint.

Diagnosis:
1. Stage 0 pinned the goal to the configured corner (18, 18). ~90% of the run
   taught "drive the flock to the top-right". Validation always randomizes the
   goal, so 0.000 was the correct score by construction.
2. The stage was recomputed from scratch every step, with no hysteresis. It
   flipped 0.00 ↔ 0.33 repeatedly, eventually within <1k steps.
3. Stage-0.33 collision ceiling 10.0 sat below the policy's measured 10.5–13.

Train-side trend: SR fell 0.28 (30k) → 0.06 (227k); FracGoal peaked 0.38 (135k);
policy std decayed 1.00 → 0.74.

Fixes (commit `621fb75`): goal randomized from stage 0; `_resolve_stage()` with
`min_dwell_steps` 25k + `demote_margin` 0.25; collision ceilings 10/8/6 → 14/11/9.

Raw log: not retained (was in `/tmp`).

---

## Run 2 — curriculum-fix run (stopped at 150k), 2026-09-23

| | |
|---|---|
| Command | same as run 1 |
| Code | `621fb75` |
| Budget | 1M planned, stopped at ~150k |
| Throughput | ~172 fps |
| Raw log | `logs/runs/run2_v3_seed0_aborted150k.log` |

**Outcome: fix 1 confirmed, stopped over a new curriculum stall.**

Validation (12 episodes each):

| Step | SR | FracGoal | MeanDist | Stage |
|---|---|---|---|---|
| 25k | 0.083 | 0.206 | 6.63 | 0.00 |
| 50k | 0.167 | 0.250 | 4.01 | 0.00 |
| 75k | 0.333 | 0.409 | 3.76 | 0.00 |
| 100k | 0.250 | 0.306 | 5.64 | 0.00 |
| 125k | 0.417 | **0.522** | 2.50 | 0.33 (promoted @120,372) |
| 150k | 0.250 | 0.268 | 3.96 | 0.00 (demoted @145,380) |

Training:

| Step | SR | FracGoal | Vis | Prog | CollEvt | Stage |
|---|---|---|---|---|---|---|
| 31k | 0.027 | 0.103 | 0.40 | 0.020 | 44.7 | 0.00 |
| 55k | 0.190 | 0.308 | 0.46 | 0.035 | 39.0 | 0.00 |
| 80k | 0.410 | 0.555 | 0.53 | 0.062 | 27.5 | 0.00 |
| 104k | 0.460 | 0.610 | 0.57 | 0.073 | 18.9 | 0.00 |
| 129k | 0.450 | 0.620 | 0.62 | 0.074 | 17.7 | 0.33 |
| 154k | 0.470 | 0.630 | 0.65 | 0.084 | 16.9 | 0.00 |

Findings:
- Goal randomization from stage 0 works: validation left 0.000 at the very first
  checkpoint, versus 17 consecutive zeros in run 1.
- The collision gate was the only binding constraint. At 145k, FracGoal 0.62,
  visibility 0.64 and progress 0.079 already cleared the **stage-0.66** gates
  (0.55 / 0.50 / -0.005), but CollEvt ~18 exceeded the stage-0.33 demotion
  bound (14 × 1.25 = 17.5).
- Cause: ceilings were calibrated on run 1's fixed-corner goal. With random goals
  the flock is often driven toward a wall, and wall contact counts as a collision.
  CollEvt plateaued at 17–18 rather than 10–13.
- Hysteresis worked as designed: one promotion and one demotion in 150k steps,
  25k apart, versus sub-1k flapping in run 1. The stage was still going to cycle
  every ~25k and never reach breadth ≥ 0.66.

Fix: collision ceilings 14/11/9 → **22/19/16** in `configs/research/v3.yaml`
(same tightening gradient, above the measured plateau). Tests: 47/47 pass.

Caveat for the paper: this loosens what the curriculum demands rather than
reducing collisions. Report CollEvt alongside success, not only the stage reached.

---

## Run 3 — recalibrated collision ceilings (stopped at 160k), 2026-09-23

| | |
|---|---|
| Command | same as run 1 |
| Code | `621fb75` + collision ceilings 22/19/16 |
| Budget | 1M planned, stopped at ~160k |
| Throughput | ~240 fps |
| Raw log | `logs/runs/run3_v3_seed0_aborted160k.log` |

**Outcome: stopped. The curriculum oscillated with the dwell period.**

Validation:

| Step | SR | FracGoal | MeanDist | Stage |
|---|---|---|---|---|
| 25k | 0.000 | 0.000 | 9.09 | 0.00 |
| 50k | 0.083 | 0.129 | 5.60 | 0.00 |
| 75k | 0.167 | 0.263 | 4.98 | 0.00 |
| 100k | 0.167 | **0.391** | 3.69 | 0.33 (promoted @83,820) |
| 125k | 0.083 | 0.151 | 4.69 | 0.00 (demoted @108,828) |
| 150k | 0.333 | 0.339 | 4.60 | 0.33 (promoted @133,920) |
| — | | | | 0.00 (demoted @158,928) |

Training:

| Step | SR | FracGoal | Vis | Prog | CollEvt | Stage |
|---|---|---|---|---|---|---|
| 18k | 0.040 | 0.092 | 0.42 | 0.022 | 40.5 | 0.00 |
| 43k | 0.098 | 0.149 | 0.45 | 0.033 | 41.3 | 0.00 |
| 68k | 0.152 | 0.220 | 0.49 | 0.040 | 38.6 | 0.00 |
| 92k | 0.280 | 0.392 | 0.62 | 0.060 | 30.6 | 0.33 |
| 117k | 0.330 | 0.479 | 0.65 | 0.076 | 24.7 | 0.00 |
| 141k | 0.340 | 0.516 | 0.62 | 0.074 | 24.9 | 0.33 |
| 160k | 0.380 | 0.533 | 0.65 | 0.080 | 27.0 | 0.30 |

Findings:
- Same seed and config as run 2 apart from the ceilings, but slower early learning
  (train FracGoal 0.30 vs 0.52 at ~75k). Subprocess envs are not bit-reproducible,
  so single-seed runs diverge. That argues for multiple seeds in the paper.
- Demoted twice, each time just after the 25k dwell expired, **while every task
  metric kept improving** (FracGoal 0.39 → 0.53, SR 0.28 → 0.38).
- Mechanism: promotion widens the layouts (adds gates, raises breadth
  0.25 → 0.6), and that alone raises collisions (24.6 → 27.0 within one stage-0.33
  stint). That crosses the demotion bound 22 × 1.25 = 27.5 on the 40-episode window.
  The collision gate is therefore self-defeating as a demotion criterion: the
  promotion causes the violation. Raising the ceiling again would only move the loop.

Fix: new curriculum option `demote_on_collisions` (default `true`, backwards
compatible), set to `false` in `configs/research/v3.yaml`. Collisions still gate
**promotion**; demotion now responds only to delivery, success, visibility and
progress. Collisions remain penalized in the reward. New test
`test_collisions_gate_promotion_only_when_demotion_on_them_is_off`; 48/48 pass.

---

## Run 4 — collision gate promotion-only (stopped at 350k), 2026-09-23

| | |
|---|---|
| Command | same as run 1 |
| Code | `621fb75` + ceilings 22/19/16 + `demote_on_collisions: false` |
| Budget | 1M planned, stopped at ~350k |
| Throughput | ~240 fps |
| Raw log | `logs/runs/run4_v3_seed0_aborted350k.log` |

**Outcome: the curriculum fix worked. Stopped over a checkpoint bug and late drift.**

Validation (12 episodes each):

| Step | SR | FracGoal | MeanDist | Stage events |
|---|---|---|---|---|
| 25k | 0.083 | 0.160 | 6.62 | |
| 50k | 0.250 | 0.348 | 5.72 | → 0.33 @50,004 |
| 75k | 0.167 | 0.219 | 5.26 | |
| 100k | 0.167 | 0.248 | 4.34 | |
| 125k | 0.167 | 0.274 | 3.74 | |
| 150k | 0.500 | **0.592** | 2.74 | |
| 175k | 0.167 | 0.410 | 3.09 | → 0.66 @172,044 (first time any run passed 0.33) |
| 200k | 0.333 | 0.411 | 3.45 | → 0.33 @197,052 |
| 225k | 0.417 | 0.479 | 3.09 | → 0.66 @235,212 |
| 250k | 0.250 | 0.351 | 4.34 | → 0.33 @264,300 |
| 275k | 0.333 | 0.420 | 4.06 | → 0.66 @291,348 |
| 300k | 0.333 | 0.406 | 4.05 | → 0.33 @316,356 |
| 325k | 0.167 | 0.407 | 3.62 | |
| 350k | 0.250 | 0.298 | 4.21 | |

Training:

| Step | SR | FracGoal | Vis | Prog | CollEvt | Stage |
|---|---|---|---|---|---|---|
| 49k | 0.333 | 0.387 | 0.54 | 0.045 | 28.5 | 0.00 |
| 98k | 0.360 | 0.505 | 0.68 | 0.064 | 21.1 | 0.33 |
| 147k | 0.360 | 0.486 | 0.68 | 0.072 | 17.3 | 0.33 |
| 197k | 0.340 | 0.460 | 0.67 | 0.065 | 15.0 | 0.66 |
| 246k | 0.370 | 0.478 | 0.69 | 0.070 | 14.9 | 0.66 |
| 295k | 0.360 | 0.473 | 0.66 | 0.074 | 15.1 | 0.56 |
| 344k | 0.280 | 0.393 | 0.61 | 0.058 | 14.1 | 0.33 |

Policy std decayed 0.95 → 0.71; explained variance ~0.86 (PPO itself healthy).

Findings:
- Fastest curriculum so far: stage 0.33 at 50k, stage 0.66 at 172k. Collisions fell
  to ~13–15 on their own once they no longer caused demotions. That is **below**
  the old ceilings, so relaxing the gate did not cost collision performance.
- From 172k the stage alternated 0.33 ↔ 0.66 with a ~30k period. These demotions
  are driven by **delivery** (at full breadth, train FracGoal fell ~0.53 → 0.46),
  which is the intended behaviour. Stage 1.0 (FracGoal ≥ 0.70) was never in reach.
- Validation peaked at 150k (0.592) and then held ~0.35–0.48. Training delivery
  plateaued at ~0.47–0.52 from ~100k, then drifted down to 0.38–0.39 by 350k,
  even at stage 0.33, while collisions kept falling. A plausible reading: late in
  the run the policy trades delivery near walls for fewer wall contacts.
- **Bug found:** `GeneralizationEvalCallback` saved `_best.zip` without the
  VecNormalize statistics of that moment. Only the end-of-run statistics were
  written, so the best checkpoint could only be evaluated with mismatched
  normalization. This affects every earlier run's `_best.zip` too.
- 12 validation episodes give SE ≈ 0.1 on FracGoal, so best-checkpoint selection
  mostly picks the luckiest draw (e.g. 0.592 at 150k vs 0.27 and 0.41 on either side).

Fixes for run 5:
1. `_best.zip` now saves `_best_vecnormalize.pkl` at the same instant; the path is
   recorded as `validation.best_vecnormalize_path` in the metadata.
2. Validation episodes 12 → 30 (SE ≈ 0.06).
3. `lr_schedule: linear` (new option in `models.py`): learning rate anneals
   3e-4 → 0 over the run, as in the original PPO, so the policy settles instead of drifting.
4. Budget 1M → 600k: every run peaked by ~150–250k.

Tests: 50/50 (new `tests/test_training_artifacts.py` for the LR schedule), plus an
end-to-end smoke run confirming both `_best` files are written and reload.

---

## Run 5 — paired best-checkpoint stats, LR annealing, 600k (completed), 2026-09-23

| | |
|---|---|
| Command | same as run 1 |
| Code | run-4 code + best-VecNormalize fix + `lr_schedule: linear` |
| Config | `configs/research/v3.yaml`: 600k steps, 30 validation episodes |
| Wall time | ~75 min (slowed by concurrent BC eval; ~140 fps average) |
| Artifacts | `models/research_v3/recurrent/recurrent_seed0{,_best}.zip` + matching `*_vecnormalize.pkl`, `recurrent_seed0_metadata.json` |
| Raw log | `logs/runs/run5_v3_seed0.log` |

**Outcome: completed. This is the model reported in the paper.**

Curriculum: stage 0.33 @104,244 → 0.66 @177,720 → 0.33 @202,728 → 0.66 @474,708
→ 0.33 @499,716 → 0.66 @565,104 → 0.33 @590,112. Time per stage: 0.00 17.3%,
0.33 70.2%, 0.66 12.5%, 1.0 never reached (training printed the built-in warning).

Validation (30 episodes each; full breadth, randomized flock size):

| Step | SR | FracGoal | MeanDist |
|---|---|---|---|
| 25k | 0.100 | 0.134 | 7.54 |
| 50k | 0.067 | 0.088 | 6.61 |
| 75k | 0.033 | 0.092 | 6.22 |
| 100k | 0.133 | 0.205 | 5.14 |
| 150k | 0.167 | 0.234 | 4.75 |
| 200k | 0.067 | 0.147 | 4.89 |
| 250k | 0.200 | 0.247 | 4.49 |
| 300k | 0.167 | 0.256 | 4.70 |
| 325k | 0.233 | 0.322 | 3.96 |
| 350k | 0.167 | 0.243 | 4.53 |
| 400k | 0.167 | 0.271 | 4.23 |
| 450k | 0.333 | 0.382 | 3.76 |
| 500k | 0.233 | 0.294 | 4.28 |
| 550k | 0.267 | 0.338 | 4.06 |
| 575k | 0.200 | 0.330 | 3.71 |
| 600k | 0.300 | **0.410** | 3.34 |

Training:

| Step | SR | FracGoal | Vis | Prog | CollEvt | Stage |
|---|---|---|---|---|---|---|
| 61k | 0.302 | 0.400 | 0.46 | 0.057 | 35.9 | 0.00 |
| 123k | 0.480 | 0.580 | 0.59 | 0.087 | 25.0 | 0.33 |
| 184k | 0.400 | 0.525 | 0.66 | 0.078 | 24.0 | 0.56 |
| 246k | 0.230 | 0.361 | 0.63 | 0.060 | 24.8 | 0.33 |
| 307k | 0.300 | 0.383 | 0.67 | 0.064 | 23.9 | 0.33 |
| 369k | 0.240 | 0.428 | 0.66 | 0.061 | 22.0 | 0.33 |
| 430k | 0.270 | 0.431 | 0.65 | 0.075 | 16.0 | 0.33 |
| 492k | 0.260 | 0.439 | 0.65 | 0.080 | 13.7 | 0.66 |
| 553k | 0.330 | 0.494 | 0.65 | 0.075 | 17.7 | 0.33 |

Final policy std 0.70.

Findings:
- With 30 episodes the validation curve is far smoother than runs 2–4, and it
  **rises through the second half** (0.24 @350k → 0.41 @600k) as the LR anneals.
  That is the opposite of run 4's late decline. The best checkpoint is the final one.
- Collisions fell 36 → 14–18 without being a demotion criterion.
- The 0.33 ↔ 0.66 alternation persisted: the policy can't hold FracGoal above the
  stage-0.66 demotion bound (0.41) at full breadth. Stage 1.0 (FracGoal ≥ 0.70) was
  out of reach, so the run trained mostly at breadth 0.6.
- Diagnostic on the 325k best (50 episodes of `test_procedural`): FracGoal 0.139
  with randomized flock size vs 0.226 with a fixed 10-sheep flock, so flock-size
  variation accounts for part of the validation/test difficulty.

---

## Baseline evaluations on the held-out suite

Script: `scripts/evaluate_generalization.py`, extended on 2026-09-23 to accept
`--model-type heuristic` and `--fixed-sheep-count`. Heuristic and BC run with
`observation_mode: legacy` and a fixed 10-sheep flock, because they decode the
legacy vector. Everything else (goals, layouts, dynamics, visibility) is randomized
exactly as for RL; episode seeds start at 100000 for every agent.

### Cluster-aware heuristic (150 episodes/scenario)

`uv run python scripts/evaluate_generalization.py --model-type heuristic --episodes 150 --output-dir results/generalization_v3/heuristic`

| Split | Scenario | SR | FracGoal [95% CI] | MeanDist | Ep. len |
|---|---|---|---|---|---|
| test | test_procedural | 0.353 | 0.468 [0.395, 0.537] | 3.44 | 532 |
| unseen | corridor | 0.000 | 0.000 [0.000, 0.000] | 12.21 | 700 |
| unseen | dense | 0.193 | 0.269 [0.207, 0.333] | 6.41 | 634 |
| unseen | narrow_gate | 0.000 | 0.003 [0.000, 0.007] | 6.75 | 700 |
| unseen | open_field | 0.980 | 0.989 [0.975, 1.000] | 1.21 | 191 |
| unseen | split_field | 0.080 | 0.317 [0.276, 0.360] | 3.97 | 660 |

### Behavioral cloning (random forest, 100 episodes/scenario)

Model `models/imitation/random_forest/behavioral_cloning.pkl` (300 trees, depth 18).
Run as six parallel single-scenario processes, merged into
`results/generalization_v3/behavioral_cloning/generalization_report.csv`:

```bash
for sc in test_procedural unseen_corridor unseen_dense unseen_narrow_gate unseen_open_field unseen_split_field; do
  uv run python scripts/evaluate_generalization.py --model-type behavioral_cloning \
    --model-path models/imitation/random_forest/behavioral_cloning.pkl --run-name behavioral_cloning \
    --episodes 100 --scenarios $sc --output-dir results/generalization_v3/behavioral_cloning/$sc &
done
```

| Split | Scenario | SR | FracGoal [95% CI] | MeanDist | Ep. len |
|---|---|---|---|---|---|
| test | test_procedural | 0.000 | 0.010 [0.001, 0.024] | 12.20 | 700 |
| unseen | corridor | 0.000 | 0.008 [0.000, 0.023] | 10.27 | 700 |
| unseen | dense | 0.000 | 0.062 [0.036, 0.090] | 7.83 | 700 |
| unseen | narrow_gate | 0.000 | 0.021 [0.008, 0.036] | 5.36 | 700 |
| unseen | open_field | 0.390 | 0.599 [0.519, 0.674] | 2.17 | 561 |
| unseen | split_field | 0.010 | 0.196 [0.161, 0.233] | 3.86 | 695 |

Notes:
- The first attempt (150 episodes, one process) was killed after ~2 h. The forest
  was loaded with `n_jobs=-1`, so every single-observation `predict` spun up a
  worker pool (66 ms/step vs 22 ms with `n_jobs=1`). The loader
  `load_behavioral_cloning_agent` now forces `n_jobs=1`.
- The clone collapses on the procedural suite (FracGoal 0.01 vs the expert's
  0.47). The forest dates from 2026-04-19, before the v3 generalization changes
  (`d3730e2`), when training goals were confined to the far quadrant (paper, Sec.
  "Adaptive curriculum"). Its demonstration CSV is no longer on disk, so the
  goal coverage can't be re-checked. Only open field, the preset closest to its
  data, keeps partial competence (the expert gets 0.99 there). Same qualitative story as the
  original paper: the clone copies local steering, not the expert's global logic.

### Recurrent PPO, run 5 best checkpoint (150 episodes/scenario)

Model `models/research_v3/recurrent/recurrent_seed0_best.zip` (600k, validation
FracGoal 0.410) with its paired `recurrent_seed0_best_vecnormalize.pkl`. Evaluated
twice: with the config's randomized flock size (`rppo_run5_rand`) and with
`--fixed-sheep-count` (10 sheep, `rppo_run5_fixed`), which matches the baselines
episode for episode. Six parallel single-scenario processes per variant, merged into
`results/generalization_v3/rppo_run5_{rand,fixed}/generalization_report.csv`.
The presets always use the base flock size, so the two variants differ only on
`test_procedural`.

| Split | Scenario | SR | FracGoal [95% CI] | MeanDist | Ep. len |
|---|---|---|---|---|---|
| test | test_procedural (random flock size) | 0.120 | 0.209 [0.155, 0.264] | 5.22 | 647 |
| test | test_procedural (10 sheep) | 0.160 | 0.261 [0.200, 0.322] | 4.28 | 623 |
| unseen | corridor | 0.000 | 0.007 [0.000, 0.017] | 11.58 | 700 |
| unseen | dense | 0.107 | 0.289 [0.236, 0.345] | 3.85 | 673 |
| unseen | narrow_gate | 0.000 | 0.005 [0.000, 0.015] | 10.68 | 700 |
| unseen | open_field | 0.487 | 0.565 [0.490, 0.637] | 3.09 | 557 |
| unseen | split_field | 0.453 | 0.576 [0.512, 0.644] | 2.21 | 483 |

## Head-to-head (identical episodes, 10-sheep flock)

FracGoal [95% CI] / SR. Heuristic 150 episodes, BC 100, RL 150.

| Scenario | Heuristic | BC | Recurrent PPO (run 5) |
|---|---|---|---|
| test_procedural | **0.468** [0.395, 0.537] / 0.35 | 0.010 / 0.00 | 0.261 [0.200, 0.322] / 0.16 |
| split_field | 0.317 [0.276, 0.360] / 0.08 | 0.196 / 0.01 | **0.576** [0.512, 0.644] / **0.45** |
| dense | 0.269 [0.207, 0.333] / **0.19** | 0.062 / 0.00 | 0.289 [0.236, 0.345] / 0.11 |
| open_field | **0.989** [0.975, 1.000] / 0.98 | 0.599 / 0.39 | 0.565 [0.490, 0.637] / 0.49 |
| corridor | 0.000 / 0.00 | 0.008 / 0.00 | 0.007 / 0.00 |
| narrow_gate | 0.003 / 0.00 | 0.021 / 0.00 | 0.005 / 0.00 |

Reading:
- Versus the original paper (structured RL: 0% on split and open field), RL now
  delivers on held-out geometry: **45% success on split field and 49% on open field**.
- RL **beats the heuristic on split field** (non-overlapping CIs; SR 0.45 vs 0.08).
  That is the fragmented-flock case the heuristic's single collect/drive switch
  handles badly. Dense is a tie on FracGoal (CIs overlap).
- The heuristic still wins clearly on the procedural suite (0.47 vs 0.26) and on
  open field (0.99 vs 0.57).
- Corridor and narrow gate defeat every agent: both need the flock threaded
  through a passage, and nothing in training rewards that explicitly.
- Single seed. The seed-to-seed variance seen across runs 2–5 means these RL
  numbers need 2+ more seeds before they are a claim rather than a data point.

---

## Runs 6 and 7 — extra seeds of the run-5 config (completed), 2026-09-24

| | |
|---|---|
| Commands | `.venv/bin/python scripts/train_v3_recurrent.py --config configs/research/v3.yaml --seed 1` (run 6), then `--seed 2` (run 7) |
| Code | `6539378` (identical to run 5) |
| Purpose | Seed variance for the paper. Run 5 is a single seed, and runs 2 and 3 showed large run-to-run differences |
| Scheduling | Sequential. One run uses ~4.8 GB of the 7.5 GB RAM, so two in parallel risked OOM |
| Artifacts | `models/research_v3/recurrent/recurrent_seed{1,2}{,_best}.zip` + `*_vecnormalize.pkl` + metadata |
| Raw logs | `logs/runs/run6_v3_seed1.log`, `logs/runs/run7_v3_seed2.log` |

Validation FracGoal (30 episodes, every 25k):

| Step | Seed 0 (run 5) | Seed 1 (run 6) | Seed 2 (run 7) |
|---|---|---|---|
| 100k | 0.205 | 0.242 | 0.306 |
| 200k | 0.147 | 0.315 | 0.258 |
| 300k | 0.256 | 0.423 | 0.261 |
| 400k | 0.271 | 0.328 | 0.289 |
| 500k | 0.294 | 0.409 | 0.268 |
| 600k | 0.410 | 0.455 | 0.288 |
| **Best** | **0.410 @600k** | **0.455 @600k** | **0.378 @175k** |

Time per curriculum stage (0.00 / 0.33 / 0.66):

| Seed | Stage 0.00 | Stage 0.33 | Stage 0.66 | Stage 0.66 changes |
|---|---|---|---|---|
| 0 | 17% | 70% | 13% | 3 promotions, 3 demotions |
| 1 | 16% | 56% | 28% | 5 promotions, 5 demotions |
| 2 | 16% | 45% | 39% | 7 promotions, 7 demotions |

No seed reached stage 1.0.

Final training window (≈602k):

| Seed | SR | FracGoal | Vis | CollEvt |
|---|---|---|---|---|
| 1 | 0.41 | 0.53 | 0.72 | 14.0 |
| 2 | 0.30 | 0.42 | 0.67 | 14.2 |

Findings:
- Seed 1 behaves like seed 0 and ends a little higher: validation rises through the
  second half as the LR anneals, and the best checkpoint is the final one.
- Seed 2 learns early (0.38 at 175k), then stays flat at ~0.21–0.36. Its best
  checkpoint is the 175k one, before most of the full-breadth training. Seed-to-seed
  spread of the best validation score is 0.38–0.46.
- All three seeds alternate 0.33 ↔ 0.66 on the delivery bound. Seed 2 spent the
  most time at 0.66 (39%) and has the flattest curve, so time at full breadth
  alone does not predict a better policy.
- Collisions converge to ~14 per episode in every seed.

Held-out eval: see "Recurrent PPO, seeds 1 and 2" and "Three-seed summary" below.

## Recurrent PPO, seeds 1 and 2 (150 episodes/scenario), 2026-09-24

Each seed's `_best.zip` with its paired `_best_vecnormalize.pkl`. Seed 1's best is its
600k checkpoint (validation 0.455); seed 2's is its 175k checkpoint (0.378). Same
protocol as run 5: `--fixed-sheep-count` on all six scenarios plus a
randomized-flock `test_procedural` run, merged into
`results/generalization_v3/rppo_seed{1,2}_{fixed,rand}/generalization_report.csv`.
Launcher: 14 parallel single-scenario processes (see `logs/eval/`).

Incident: the first launch ran 14 processes with PyTorch's default thread pool (~39
threads each). Load average hit ~79 on 16 cores and nothing finished in 36 min.
Killed and relaunched with `OMP_NUM_THREADS=1`; everything finished in ~25–30 min.
`scripts/evaluate_generalization.py` now calls `torch.set_num_threads(1)`.

FracGoal [95% CI] / SR / MeanDist, 10 sheep:

| Scenario | Seed 1 | Seed 2 |
|---|---|---|
| test_procedural | 0.302 [0.239, 0.368] / 0.21 / 4.52 | 0.326 [0.259, 0.392] / 0.22 / 4.62 |
| split_field | 0.511 [0.444, 0.579] / 0.37 / 2.81 | 0.184 [0.148, 0.225] / 0.03 / 4.55 |
| dense | 0.286 [0.240, 0.337] / 0.08 / 3.87 | 0.040 [0.023, 0.061] / 0.01 / 5.92 |
| open_field | 0.311 [0.260, 0.366] / 0.11 / 2.93 | 0.233 [0.193, 0.274] / 0.03 / 3.05 |
| corridor | 0.005 [0.001, 0.009] / 0.00 / 10.98 | 0.000 / 0.00 / 13.19 |
| narrow_gate | 0.000 / 0.00 / 10.98 | 0.011 [0.003, 0.021] / 0.00 / 8.70 |
| test_procedural, random flock | 0.281 [0.221, 0.343] / 0.15 | 0.212 [0.155, 0.269] / 0.13 |

## Three-seed summary (goes into paper Table I)

FracGoal mean (min, max over seeds) / mean SR:

| Scenario | Heuristic | BC | Recurrent PPO, 3 seeds |
|---|---|---|---|
| test_procedural | **0.468** / 0.35 | 0.010 / 0.00 | 0.296 (0.261, 0.326) / 0.20 |
| split_field | 0.317 / 0.08 | 0.196 / 0.01 | 0.424 (0.184, 0.576) / 0.28 |
| dense | 0.269 / 0.19 | 0.062 / 0.00 | 0.205 (0.040, 0.289) / 0.06 |
| open_field | **0.989** / 0.98 | 0.599 / 0.39 | 0.370 (0.233, 0.565) / 0.21 |
| corridor | 0.000 / 0.00 | 0.008 / 0.00 | 0.004 (0.000, 0.007) / 0.00 |
| narrow_gate | 0.003 / 0.00 | 0.021 / 0.00 | 0.005 (0.000, 0.011) / 0.00 |
| test_procedural, random flock | | | 0.234 (0.209, 0.281) / 0.13 |

Reading, and what changed from the single-seed picture:
- **Procedural test is stable across seeds** (0.26–0.33, all CIs below the
  heuristic's 0.395 lower bound). The claim "heuristic > RL on the test suite" holds.
- **Presets are not.** Seed 0 was the lucky seed on open field (0.57 vs 0.31, 0.23).
  "RL beats the heuristic on split field" holds for 2 of 3 seeds (0.58, 0.51 vs 0.32,
  non-overlapping CIs); seed 2 gets 0.18.
- Seed 2 is best on the procedural test and worst on split, dense and open field. Its
  checkpoint was selected at 175k, so validation (procedural, full breadth) picked a
  policy that is good on average and brittle on specific maps.
- Single-seed headline "45% / 49% success on split / open field" becomes mean
  28% / 21% (best seed 45% / 49%). The paper now reports the means.

## Runs 8–10 — stage-0.66 gate ablation, seeds 0–2 (completed), 2026-09-24

| | |
|---|---|
| Question | Does keeping the agent at full breadth (stage 0.66) improve held-out performance? |
| Config | `configs/research/v3_gate045.yaml` = `v3.yaml` with stage-0.66 `min_fraction_at_goal` 0.55 → **0.45** (demotion bound 0.41 → ~0.34), `save_dir: models/research_v3_gate045`. Nothing else changed |
| Code | `6e7c4e1` |
| Pipeline | `logs/gate045_pipeline.sh` trains seeds 0, 1, 2 one after another (08:22–12:02 UTC, ~70–78 min each), then evaluates every `_best` exactly like runs 5–7 (12:02–12:15, 21 single-thread processes) into `results/generalization_v3/gate045_seed{0,1,2}_{fixed,rand}` |
| Raw logs | `logs/runs/run{8,9,10}_gate045_seed{0,1,2}.log`, `logs/gate045_pipeline.out`, `logs/eval/gate045_*.log` |

Note: the launcher sets an empty `OMP_NUM_THREADS` for training, so libgomp prints a
warning and falls back to its default (the same thread count as runs 5–7).

### Training

| Seed | Best validation | At step | Stage 0 / 0.33 / 0.66 | Stage changes | Longest stint at 0.66 |
|---|---|---|---|---|---|
| 0 | 0.434 | 500k | 15% / 42% / 43% | 18 | ~40k |
| 1 | 0.418 | 575k | 13% / 37% / 50% | 8 | 212k (270k–482k) |
| 2 | 0.451 | 425k | 17% / 28% / 55% | 8 | 167k (194k–361k) |
| *base 0 / 1 / 2* | *0.410 / 0.455 / 0.378* | *600k / 600k / 175k* | *13% / 28% / 39% at 0.66* | *6 / 10 / 14* | |

- The gate change did what it was meant to: 43–55% of training at full breadth,
  versus 13–39%. Seeds 1 and 2 held stage 0.66 for long stretches. Seed 0 still
  cycled every ~25–40k, because a 40-episode window right after promotion can dip
  under the 0.34 bound even though the longer training average stays ~0.37.
- No seed reached stage 1.0 (needs FracGoal ≥ 0.70 on the training window).
- All three best checkpoints come from late in training (425k–575k). Baseline seed 2's
  175k checkpoint was the outlier that hurt it on the presets.
- Final training window at stage 0.66 (full breadth): FracGoal 0.42 / 0.50 / 0.45,
  CollEvt 15–17 (slightly above the base runs' ~14, as expected with corridors in the mix).

### Held-out results (150 episodes/scenario, 10 sheep unless noted)

Per seed FracGoal, and 3-seed mean (SR mean):

| Scenario | Gate s0 | Gate s1 | Gate s2 | **Gate mean** | Base mean | Heuristic |
|---|---|---|---|---|---|---|
| test_procedural | 0.326 | 0.358 | 0.279 | **0.321** (SR 0.19) | 0.296 (0.20) | 0.468 (0.35) |
| test_procedural, random flock | 0.279 | 0.284 | 0.305 | **0.289** (SR 0.16) | 0.234 (0.13) | |
| split_field | 0.340 | 0.356 | 0.218 | 0.305 (SR 0.12) | **0.424** (0.28) | 0.317 (0.08) |
| dense | 0.385 | 0.401 | 0.179 | **0.322** (SR 0.13) | 0.205 (0.06) | 0.269 (0.19) |
| open_field | 0.627 | 0.473 | 0.346 | **0.482** (SR 0.28) | 0.370 (0.21) | 0.989 (0.98) |
| corridor | 0.036 | 0.064 | 0.021 | **0.040** (SR 0.02) | 0.004 (0.00) | 0.000 |
| narrow_gate | 0.052 | 0.052 | 0.082 | **0.062** (SR 0.02) | 0.005 (0.00) | 0.003 |
| mean of the 6 held-out | | | | **0.255** | 0.217 | 0.341 |

Per-seed 95% CIs are in the merged CSVs.

Findings:
- **The gate change improves the mean on 5 of 6 held-out scenarios**, and the
  randomized-flock test (+5.5 points). The largest gains are where breadth-0.6
  training had no coverage: corridors appear only at stage ≥ 0.66, and so do
  randomized flock sizes. **Corridor and narrow gate are above zero for every gate
  seed** (CIs exclude 0, every gate seed > every base seed), though still tiny (2–8%).
- **Split field got worse** (0.42 → 0.31). Both base seeds that beat the heuristic
  there (0.58, 0.51) have no counterpart; all three gate seeds sit at or below the
  heuristic (0.317). Training time moved away from breadth 0.6, where bar layouts are
  a larger share (bars 0.25 at stage 0.33 vs 0.20 at full breadth), and toward
  corridors and gates.
- **Seeds agree more.** The largest per-scenario spread across seeds is 28 points
  (open field) versus 39 for the base config (split field). On the procedural test the
  spread is 8 points (0.28–0.36).
- **The procedural test gain is small** (0.296 → 0.321), inside the seed spread of both
  arms. The heuristic's 0.468 is still well above every seed.
- Net: longer training at full breadth trades specialization (split field) for
  coverage (corridors, gates, flock sizes, open field). Three seeds per arm can show
  the large effects (corridor/gate, split field) but not the small one on the test suite.

## Legacy versions v1 and v2 (150 episodes per setting), 2026-09-24

For the paper's development-history section. Script:
`scripts/evaluate_legacy_versions.py --version {v1,v2}` → `results/legacy/{v1,v2}_report.csv`.

- **v1** = `HerdingEnv-v0`: full visibility, no obstacles, 10 sheep scattered in
  [5, 15]², goal fixed at (18, 18), 500 steps. Model `models/legacy_v1/ppo_herding.zip`,
  restored from the first commit (`9d8d987`; it was deleted from the tree in `f764238`).
- **v2** = `HerdingEnv-v2`: visibility radius 8 with the 999 sentinel, two default
  obstacles, goal fixed at (18, 18), 600 steps. Model `scripts/models/ppo_herding_v2.zip`.
- **Neither observation contains the goal.** v1 is (dog position, sheep relative to
  dog); v2 adds the visibility mask and obstacle boxes. A policy can only learn "herd
  to (18, 18)".

Settings: PPO on its training setting; PPO with the goal drawn uniformly from
[2.4, 17.6]² (v3's goal region); a seeded uniform-random policy on the training setting.
Seeds 100000+.

| Version | Agent | Goal | SR | FracGoal [95% CI] | MeanDist | Ep. len |
|---|---|---|---|---|---|---|
| v1 | PPO | fixed (18, 18) | 0.993 | 0.993 [0.980, 1.000] | 1.22 | 92 |
| v1 | PPO | random | 0.020 | 0.023 [0.003, 0.047] | 12.21 | 491 |
| v1 | random policy | fixed | 0.040 | 0.056 [0.025, 0.091] | 12.12 | 490 |
| v2 | PPO | fixed (18, 18) | 0.693 | 0.715 [0.641, 0.781] | 3.62 | 360 |
| v2 | PPO | random | 0.033 | 0.057 [0.027, 0.089] | 9.53 | 588 |
| v2 | random policy | fixed | 0.133 | 0.137 [0.087, 0.193] | 11.38 | 576 |

Reading: both versions solve their own fixed-corner task (99% and 69% success) and
drop to the level of a random policy or below when the goal moves. That's the first
link in the chain v1 → v2 → structured v3 → procedural v3: each version removed one
source of overfitting (goal, then layouts) and exposed the next.

## Rollout figure

`scripts/paper_figures.py` also writes `docs/paper/figures/v3_rollouts.png` (light
mode, replacing the dark 3-D GIF stills). It shows dog and flock-centroid paths from
the saved eval trajectories (`trajectories.csv`) for the heuristic and RL seed 0 on
the same preset episode. The episode is the first one where the two agents' outcomes
differ (split field #2, open field #3), so it is illustrative, not typical.
Observation: on open field, RL brings the flock to the goal and then circles it for the
rest of the episode instead of holding it there. The heuristic finishes in 78 steps.

## Delivered-then-lost analysis (all agents, from episode summaries), 2026-09-24

`best_fraction_at_goal` = highest fraction of the flock inside the goal at any step;
FracGoal = at the final step. "Lost" = best − final. "Reached-then-lost" = share of
episodes with best ≥ 0.8 but final < 0.5. RL rows pool the 3 seeds (450 episodes).

| Scenario | Agent | Final | Best | Lost | Reached-then-lost |
|---|---|---|---|---|---|
| test_procedural | Heuristic | 0.47 | 0.63 | 0.16 | 0.18 |
| | RL base | 0.30 | 0.44 | 0.15 | 0.09 |
| | RL gate | 0.32 | 0.53 | 0.21 | 0.13 |
| split_field | Heuristic | 0.32 | 0.46 | 0.15 | 0.09 |
| | RL base | 0.42 | 0.69 | 0.27 | 0.23 |
| | RL gate | 0.30 | 0.65 | 0.35 | 0.31 |
| dense | Heuristic | 0.27 | 0.39 | 0.13 | 0.18 |
| | RL base | 0.21 | 0.51 | 0.31 | 0.22 |
| | RL gate | 0.32 | 0.70 | 0.37 | 0.29 |
| open_field | Heuristic | 0.99 | 1.00 | 0.01 | 0.01 |
| | BC | 0.60 | 0.85 | 0.26 | 0.18 |
| | RL base | 0.37 | 0.70 | 0.33 | 0.24 |
| | RL gate | 0.48 | 0.79 | 0.31 | 0.23 |

Findings:
- On the presets, RL gets 65–79% of the flock into the goal at some point but ends
  with 30–48%. That's a third of the flock lost after arrival. The heuristic loses 1%
  on open field and 13–16% elsewhere.
- In 22–31% of RL preset episodes the flock was ≥ 80% delivered and then fell
  below 50%. That matches the rollout figure (open field: flock reaches the goal,
  then the dog circles it back out).
- Gate RL reaches the goal more often (best 0.70 dense, 0.79 open) but holds it no
  better. Delivery improved, retention didn't.
- Likely cause: success needs every sheep inside at the same step, and the reward
  has no term for keeping delivered sheep inside. The progress terms reward reducing
  the mean and max distance, so they give nothing once sheep are in, while the
  drive-position term keeps pulling the dog to the far side of the flock, which pushes
  sheep through the goal. A holding reward, or ending the episode at a high fraction,
  is a direct next experiment.
- On the procedural test the gap is smaller (0.15–0.21 lost), so the preset maps
  exaggerate it.

## Paper figures and final paper (2026-09-24)

`scripts/paper_figures.py` regenerates every figure in `docs/paper/figures/` from the
eval CSVs and run logs (all light mode):
- `v3_layouts.png`: first test-split seeds (from 100000) with the blobs, corridor,
  gate and bars topologies.
- `v3_results.png`: held-out FracGoal. Heuristic, BC, RL base (runs 5–7) and RL
  gate 0.45 (runs 8–10). Baseline bars show a bootstrap CI; RL bars show the 3-seed
  mean with a min–max seed range.
- `v3_training.png`: 3-seed mean and range of validation FracGoal per curriculum, plus
  each run's share of steps per stage.
- `v3_rollouts.png`: see "Rollout figure" above.

`docs/paper/main.typ` was rewritten from scratch after runs 8–10 (not patched):
development history (v1, v2, structured v3, procedural v3), method, three-seed
results for both curricula, seed variance, curriculum ablation, delivered-but-not-held,
training stability, discussion, limitations. Six pages; no em or en dashes in the
text (the only ones are the IEEE template's "Abstract—"/"Index Terms—" and
reference page ranges). Column figures and tables use `placement: auto` so they
float to column tops instead of leaving gaps.
