#!/usr/bin/env bash
# Run wheel_bench_probe.py with the environment of probe-amp-left-yaw.sh (same checks, same DDS setup).
#   bash run_wheel_probe.sh left|right [hold|full]      (no stage = read only: no RPC, no command)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TOOLS_DIR}/../../../.." && pwd)"
wheel="${1:?usage: run_wheel_probe.sh left|right [hold|full]}"
stage="${2:-}"
[[ "${wheel}" == left || "${wheel}" == right ]] || { echo "wheel must be left or right" >&2; exit 2; }
[[ -z "${stage}" || "${stage}" == hold || "${stage}" == full ]] || { echo "stage must be hold or full" >&2; exit 2; }
export ROS_DISTRO=jazzy ROS_DOMAIN_ID=99 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp SIM2REAL_DDS_MODE=local
unset ROS_RUN_EXTRA_SETUP ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS CYCLONEDDS_URI
unset PACIFIC_RIM_NATIVE_DDS_DOMAIN_ID PACIFIC_RIM_NATIVE_DDS_DOMAIN_OFFSET
running="$(docker ps --format '{{.Names}}')"
if grep -Eq 'lhframework_hw|lhframework_runner|beyondmimic|sbus_to_cmdvel' <<<"${running}"; then
  echo "Stop the motion chain before the wheel bench test; the shared motor driver may remain running." >&2
  exit 1
fi
if ! grep -Fxq 'pacific-rim-ros2-run-motor_driver-motor_driver_node-jazzy-d99' <<<"${running}"; then
  echo "The shared domain-99 motor driver must already be running." >&2
  exit 1
fi
if [[ -n "${stage}" ]]; then
  echo "The robot must be HUNG with BOTH wheels free, the policy chain stopped, the operator at the e-stop."
  read -r -p "Type YES to drive the ${wheel} wheel (stage ${stage}): " answer
  [[ "${answer}" == YES ]] || { echo "not confirmed."; exit 1; }
fi
source "${TOOLS_DIR}/runtime-artifact-preflight.sh"
configure_formal_runtime
source "${TOOLS_DIR}/amp-dds-env.sh"
prepare_amp_dds wheel-bench
output="$(mktemp -d "${ROOT_DIR}/module/service/motion_control_sim2real_service/.cache/amp-dds/wheel-bench-XXXXXXXX")"
echo "Output: ${output}/${wheel}.jsonl"
args=(--wheel "${wheel}" --output "/workspace/${output#"${ROOT_DIR}/"}/${wheel}.jsonl")
[[ -z "${stage}" ]] || args+=(--stage "${stage}" --yes)
"${ROOT_DIR}/pr" ros2:exec --network host python3 \
  /workspace/module/service/motion_control_sim2real_service/tools/wheel_bench/wheel_bench_probe.py "${args[@]}"
