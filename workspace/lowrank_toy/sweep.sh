#!/bin/bash
# Local sweep: explore rank for uv_lowr and lambda for rand_diff.
# All on the same synthetic low-rank dataset, 2000 steps, 1 seed for speed
# (we already have 3-seed results for the headline variants).
set -e
cd "$(dirname "$0")"
mkdir -p sweep_runs

# rank sweep for uv_lowr (skip rank=4, already done)
for r in 8 16 32; do
  out="sweep_runs/uv_rank${r}.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "=== uv_lowr rank=$r ==="
  /opt/venv/bin/python3 toy_pilot.py --steps 2000 --seed 0 --ffn-rank $r --out "$out" 2>&1 | tail -3
done

# lambda sweep for rand_diff
for lam in 0.001 0.01 0.1 0.5; do
  out="sweep_runs/rand_diff_lam${lam}.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "=== rand_diff lambda=$lam ==="
  /opt/venv/bin/python3 toy_pilot.py --steps 2000 --seed 0 --reg-weight $lam --out "$out" 2>&1 | tail -3
done

echo "DONE sweep"
