#!/bin/bash
# Turning checkpoint evaluation (Exp 41 / 78 / 79): turning grid, straight-line symmetry, posture, 30 s standing.
#
# usage: tools/nemov3_tools/eval_turn.sh <task> <run dir> <tag prefix> <iteration> [iteration ...]
#
# For each model_<iteration>.pt (waits until it exists, so it can run alongside the training):
#   TURN   turn_eval.py on vx 0 / 0.6 / 1.0 / 1.3 x wz -1.3 / -0.6 / 0 / 0.6 / 1.3: turn / speed tracking, leg swing
#          while turning vs straight, left/right alternation, falls
#   SB     sym_batch_eval.py at 0.3 / 0.6 / 1.0 / 1.3 m/s: straight-line left/right symmetry
#   PB     posture_batch_eval.py at 0 / 0.6 / 1.0 / 1.3 m/s: standing still, lean, weave
#   fell   stand_fall_time.py: falls while standing for 30 s
set -u
cd "$(dirname "$0")/../.."
TASK=$1; RUN=$2; TAG=$3; shift 3
T=tools/nemov3_tools
for IT in "$@"; do
  CK=$RUN/model_$IT.pt
  until [ -f "$CK" ]; do sleep 60; done
  sleep 20
  echo "=== $TAG@$IT"
  uv run python $T/turn_eval.py "$TASK" "$CK" 32 0,0.6,1.0,1.3 -1.3,-0.6,0,0.6,1.3 600 "$TAG@$IT" 2>/dev/null | grep -E "^TURN"
  uv run python $T/sym_batch_eval.py "$TASK" "$CK" 32 0.3,0.6,1.0,1.3 750 "$TAG@$IT" 2>/dev/null | grep -E "^(SB|SYM)"
  uv run python $T/posture_batch_eval.py "$TASK" "$CK" 64 0,0.6,1.0,1.3 600 "$TAG@$IT" 2>/dev/null | grep -E "^PB"
  uv run python $T/stand_fall_time.py "$TASK" "$CK" "$TAG@$IT" 2>/dev/null | grep -vE "^\s*$" | tail -3
done 2>&1
