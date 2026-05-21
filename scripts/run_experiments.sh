#!/bin/bash
# Sequentially run the 5 ablation experiments (exp2..exp6) defined in
# scripts/exp*_*.yaml, then write a final comparison table.
#
# Each per-experiment run already blocks on `runtime.min_free_gpu_mb` via the
# pre-flight check in stage_gsplat(), so a busy GPU just stalls the queue
# until enough memory frees up (configured to 10000 MiB per the user's spec).
#
# Logs:
#   logs/queue.log           — high-level start/finish markers
#   logs/<exp-name>.log       — full stdout/stderr per experiment
#   logs/final_comparison.log — compare_runs.py output
#
# Run via Claude background task (or `nohup ... &`); the script outlives any
# transient shell session.

set -e
cd "$(dirname "$0")/.."   # repo root
mkdir -p logs

run() {
  local cfg="$1"
  local name
  name=$(basename "$cfg" .yaml)
  echo "===== [$(date)] starting $name =====" | tee -a logs/queue.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stdbuf -oL -eL \
    uv run python scripts/run_pipeline.py --config "$cfg" 2>&1 \
    | tee "logs/${name}.log"
  echo "===== [$(date)] finished $name =====" | tee -a logs/queue.log
}

run scripts/exp2_lidar_priors_var10.yaml
run scripts/exp3_imageonly_default.yaml
run scripts/exp4_imageonly_mcmc.yaml

echo "===== [$(date)] installing ppisp =====" | tee -a logs/queue.log
uv add 'ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0' 2>&1 | tee -a logs/queue.log

run scripts/exp5_imageonly_mcmc_ppisp.yaml
run scripts/exp6_full_mcmc_ppisp_var10.yaml

echo "===== [$(date)] all experiments complete =====" | tee -a logs/queue.log
uv run python scripts/compare_runs.py \
  output/sensyn_office \
  output/sensyn_office_20260521_125520 \
  output/sensyn_office_exp2_* \
  output/sensyn_office_exp3_* \
  output/sensyn_office_exp4_* \
  output/sensyn_office_exp5_* \
  output/sensyn_office_exp6_* 2>&1 | tee logs/final_comparison.log
