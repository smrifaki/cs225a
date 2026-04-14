#!/usr/bin/env bash
#
# Run all three baseline modes plus the proposed selector on the
# peg-in-hole task across the three locked seeds.
#
# Usage:
#   bash scripts/run_all_experiments.sh
#
# Assumes:
#   - The SCL world is launched separately (../sai2/openSai
#     spawns the simulator on its standard Redis ports).
#   - run_sim has been built under build/.

set -euo pipefail

SEEDS=(42 1337 2024)
MODES=(vision_only force_only always_fuse selector)
RUNS_ROOT="${RUNS_ROOT:-runs}"

mkdir -p "$RUNS_ROOT"

for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    out_dir="$RUNS_ROOT/seed_${seed}/${mode}"
    mkdir -p "$out_dir"
    echo "[run] seed=$seed mode=$mode -> $out_dir"
    ./build/run_sim \
      configs/peg_in_hole.yaml \
      --mode "$mode" \
      --seed "$seed" \
      --out-dir "$out_dir"
  done
done

python python/analysis/aggregate.py \
  --runs-root "$RUNS_ROOT" \
  --out-dir "$RUNS_ROOT/aggregated"

echo "done. results under $RUNS_ROOT/aggregated/"
