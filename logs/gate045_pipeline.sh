#!/bin/bash
# Gate ablation pipeline: train seeds 0-2 sequentially (RAM allows one run at a
# time), then evaluate each best checkpoint on the held-out suite.
cd /home/ciatel/coding/college/DS-geometric-shepherding-rl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CFG=configs/research/v3_gate045.yaml
for s in 0 1 2; do
  n=$((8 + s))
  echo "[pipeline] train seed $s start $(date -u +%H:%M)"
  OMP_NUM_THREADS= .venv/bin/python scripts/train_v3_recurrent.py --config $CFG --seed $s > logs/runs/run${n}_gate045_seed${s}.log 2>&1
  echo "[pipeline] train seed $s exit $? $(date -u +%H:%M)"
done
eval_seed() {
  s=$1; M=models/research_v3_gate045/recurrent/recurrent_seed${s}_best; V=${M}_vecnormalize.pkl
  for sc in test_procedural unseen_corridor unseen_dense unseen_narrow_gate unseen_open_field unseen_split_field; do
    .venv/bin/python scripts/evaluate_generalization.py --config $CFG --model-type recurrent --model-path $M.zip --vecnormalize $V --fixed-sheep-count --run-name gate045_seed${s}_fixed --episodes 150 --scenarios $sc --output-dir results/generalization_v3/gate045_seed${s}_fixed/$sc > logs/eval/gate045_seed${s}_fixed_$sc.log 2>&1 &
  done
  .venv/bin/python scripts/evaluate_generalization.py --config $CFG --model-type recurrent --model-path $M.zip --vecnormalize $V --run-name gate045_seed${s}_rand --episodes 150 --scenarios test_procedural --output-dir results/generalization_v3/gate045_seed${s}_rand/test_procedural > logs/eval/gate045_seed${s}_rand_test_procedural.log 2>&1 &
}
echo "[pipeline] eval start $(date -u +%H:%M)"
eval_seed 0; eval_seed 1; wait
eval_seed 2; wait
echo "[pipeline] all done $(date -u +%H:%M)"
