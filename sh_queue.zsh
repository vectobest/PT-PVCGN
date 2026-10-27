#!/bin/zsh
# Shanghai experiment queue (added 2026-10-23 at user's request): PVCGN vs PT-PVCGN, 2 seeds, 30 epochs.
# Waits until the Hangzhou queue (run_queue.sh) and any training run have finished. Skips finished runs.
cd "$(dirname "$0")"
while pgrep -f "run_queue\.sh" > /dev/null || pgrep -f "metro\.train" > /dev/null; do sleep 60; done
run() {
  local name=$1; shift
  if grep -q '"finished": true' ml_pipeline/runs/$name.json 2>/dev/null; then echo "skip $name"; return; fi
  echo "start $name $(date)"
  .venv/bin/python -m metro.train "$@" > ml_pipeline/runs/logs/$name.log 2>&1
  echo "done  $name $(date)"
}
for s in 1 2; do
  for v in pvcgn pt; do
    run sh_${v}_s$s --city sh --variant $v --seed $s --epochs 30 --batch 8 --milestones 20 25 --patience 10
  done
done
