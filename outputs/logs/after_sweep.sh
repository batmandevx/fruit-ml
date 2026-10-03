#!/bin/zsh
# Runs after phase2_sweep.sh: score the new runs on val and test, Grad-CAM checks for the variety model.
cd /Users/ayush/Downloads/date-fruit-quality-ml
until grep -q "done maturity" outputs/logs/phase2_sweep.log; do sleep 30; done
cd src
export PYTHONWARNINGS=ignore
for r in grade_effv2b0 maturity_effv2b0; do
  for s in val test; do ../.venv-metal/bin/python evaluate.py --run $r --tta --split $s 2>/dev/null | grep "^\["; done
done
../.venv-metal/bin/python evaluate.py --run grade_mnv3 grade_effv2b0 --tta --split val 2>/dev/null | grep "^\["
../.venv-metal/bin/python evaluate.py --run maturity_mnv3 maturity_effv2b0 --tta --split val 2>/dev/null | grep "^\["
../.venv-metal/bin/python explain.py --run effv2b0_260 --per-class 2 2>/dev/null | tail -12
echo AFTER_SWEEP_DONE
