#!/bin/zsh
# Stronger backbone for the Phase 2 CNNs (Metal GPU, run one at a time).
cd /Users/ayush/Downloads/date-fruit-quality-ml/src
for spec in "grade 40" "maturity 30"; do
  set -- ${=spec}
  [ -f ../outputs/models/$1_effv2b0_meta.json ] && continue
  echo "$(date +%H:%M) start $1"
  ../.venv-metal/bin/python -u train.py --data $1 --backbone efficientnetv2b0 --run $1_effv2b0 --ft-layers 0 --ft-lr 1e-4 \
    --label-smoothing 0.1 --head-epochs 15 --ft-epochs $2 > ../outputs/logs/train_$1_effv2b0.log 2>&1
  echo "$(date +%H:%M) done $1: $(tail -1 ../outputs/logs/train_$1_effv2b0.log)"
done
