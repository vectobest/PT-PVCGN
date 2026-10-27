#!/bin/zsh
# Experiment queue: Gap 1 (periodic + calendar context), Hangzhou only. Skips runs that already finished.
# Ablation: pvcgn (published model) -> p (+periodic refs) -> pt (+calendar embeddings) = PT-PVCGN.
cd "$(dirname "$0")"
# wait for a run started by a previous queue, if any
while pgrep -f "metro.train" > /dev/null; do sleep 30; done
run() {
  local name=$1; shift
  if grep -q '"finished": true' ml_pipeline/runs/$name.json 2>/dev/null; then echo "skip $name"; return; fi
  echo "start $name $(date)"
  .venv/bin/python -m metro.train "$@" > ml_pipeline/runs/logs/$name.log 2>&1
  echo "done  $name $(date)"
}
for s in 1 2 3; do
  for v in pvcgn p pt; do
    run hz_${v}_s$s --city hz --variant $v --seed $s --epochs 100 --batch 32 --milestones 50 75 --patience 20
  done
done
