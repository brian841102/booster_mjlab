#!/bin/bash
# Foot-lift checkpoint evaluation: tempo, the user's look criteria, posture.
#
# usage: tools/nemov3_tools/eval_tempo.sh <task> <run dir> <tag prefix> <iteration> [iteration ...]
#
# For each model_<iteration>.pt (waits until it exists, so it can run alongside the training):
#   LIFT   lift_eval.py at 0.8 / 1.0 / 1.3 m/s: lifts per second, height, duration, single support, micro-bounces and the
#          stride cycle; TEMPO checks the cycle against Fast06's 1.33 s +-10 % (1.20-1.46 s, user 2026-09-29)
#   CRIT   lift_pose.py at 1.0 / 1.3 m/s: the lifted foot is the rear one, stretched back, and the lifted-side hand
#          stays ahead for the whole lift
#   PB     posture_batch_eval.py at 0 / 0.6 / 1.0 / 1.3 m/s: standing still, lean, weave
set -u
cd "$(dirname "$0")/../.."
TASK=$1; RUN=$2; TAG=$3; shift 3
T=tools/nemov3_tools
for IT in "$@"; do
  CK=$RUN/model_$IT.pt
  until [ -f "$CK" ]; do sleep 60; done
  sleep 20
  echo "=== $TAG@$IT"
  # Kept in a variable, not `tee /dev/stderr`: that reopens a redirected log file at offset 0 and overwrites it.
  lift=$(uv run python $T/lift_eval.py "$TASK" "$CK" 32 0.8,1.0,1.3 0 600 "$TAG@$IT" 2>/dev/null | grep -E "^LIFT")
  echo "$lift"
  echo "$lift" \
    | awk '/vx (1.00|1.30)/ { match($0, /cycle [0-9.]+ s/); c = substr($0, RSTART + 6, RLENGTH - 8);
           ok = (c >= 1.197 && c <= 1.463) ? "ok" : "FAIL"; match($0, /vx [0-9.]+/);
           print "TEMPO " substr($0, RSTART, RLENGTH) " | stride cycle " c " s (Fast06 1.33, +-10% 1.20-1.46) " ok }'
  uv run python $T/lift_pose.py "$TASK" "$CK" 32 1.0,1.3 600 "$TAG@$IT" 2>/dev/null | grep -E "^CRIT"
  uv run python $T/posture_batch_eval.py "$TASK" "$CK" 32 0,0.6,1.0,1.3 600 "$TAG@$IT" 2>/dev/null | grep -E "^PB"
done 2>&1
