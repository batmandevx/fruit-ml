#!/bin/zsh
# CPU fallback for the Colab sweep: the two backbones that are affordable without a GPU.
cd /Users/ayush/Downloads/date-fruit-quality-ml
for spec in "mnv3_224 mobilenetv3 224 1e-4" "effv2b0_260 efficientnetv2b0 260 1e-4"; do
  set -- ${=spec}
  [ -f outputs/models/$1_meta.json ] && continue
  echo "$(date +%H:%M) start $1"
  .venv/bin/python src/train.py --backbone $2 --img-size $3 --run $1 --ft-layers 0 --ft-lr $4 \
    --label-smoothing 0.1 --head-epochs 15 --ft-epochs 40 > outputs/logs/train_$1.log 2>&1
  echo "$(date +%H:%M) done $1: $(tail -1 outputs/logs/train_$1.log)"
done
