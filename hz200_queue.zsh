#!/bin/zsh
# Equal-budget runs on Hangzhou (agreed at Gate C, 2026-10-23/24): every model with the paper's full
# 200-epoch schedule (lr x0.1 at 100 and 150). Waits for any training run. Skips finished runs.
cd "$(dirname "$0")"
while pgrep -f "sh_queue\.zsh" > /dev/null || pgrep -f "run_queue\.sh" > /dev/null || pgrep -f "metro\.train" > /dev/null; do sleep 60; done
run() {
  local name=$1; shift
  if grep -q '"finished": true' ml_pipeline/runs/$name.json 2>/dev/null; then echo "skip $name"; return; fi
  echo "start $name $(date)"
  .venv/bin/python -m metro.train "$@" > ml_pipeline/runs/logs/$name.log 2>&1
  echo "done  $name $(date)"
}
for v in pvcgn pt p; do
  for s in 1 2 3; do
    run hz_${v}_s${s}_e200 --city hz --variant $v --seed $s --epochs 200 --batch 32 --milestones 100 150 --patience 50 --tag _e200
  done
done
