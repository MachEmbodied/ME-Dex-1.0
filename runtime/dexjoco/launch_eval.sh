#!/usr/bin/env bash
# One shared policy per GPU; each server serializes concurrent client inference.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${CHECKPOINT:?Set CHECKPOINT to the policy checkpoint directory}"
: "${OUTPUT_ROOT:?Set a new OUTPUT_ROOT for this evaluation}"
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output already exists: $OUTPUT_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/logs"
read -ra tasks <<< "${TASKS:-bimanual_assembly bimanual_hanoi bimanual_microwave_cook bimanual_photograph bimanual_unlock_ipad click_mouse fold_glasses hammer_nail pick_bucket pinch_tongs water_plant}"
IFS=, read -ra gpus <<< "${GPUS:-0,1,2,3}"
base_port="${BASE_PORT:-21000}"
server_pids=(); client_pids=()
cleanup() {
  local pid
  for pid in "${client_pids[@]}" "${server_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${client_pids[@]}" "${server_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for i in "${!gpus[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" PORT="$((base_port+i))" HOST=127.0.0.1 DEVICE=cuda:0 \
    bash runtime/dexjoco/serve.sh >"$OUTPUT_ROOT/logs/server_$i.log" 2>&1 &
  server_pids+=("$!")
done
for i in "${!gpus[@]}"; do
  ready=0
  for ((attempt=0; attempt<900; attempt++)); do
    if curl -fsS "http://127.0.0.1:$((base_port+i))/health" >"$OUTPUT_ROOT/logs/health_$i.json" 2>/dev/null; then ready=1; break; fi
    kill -0 "${server_pids[$i]}" 2>/dev/null || { tail -n 50 "$OUTPUT_ROOT/logs/server_$i.log" >&2; exit 1; }
    sleep 1
  done
  [[ "$ready" == 1 ]] || { echo "Server startup timed out" >&2; exit 1; }
done
for i in "${!tasks[@]}"; do
  slot="$((i * ${#gpus[@]} / ${#tasks[@]}))"
  bash runtime/dexjoco/eval.sh "${tasks[$i]}" "http://127.0.0.1:$((base_port+slot))" \
    "$OUTPUT_ROOT/${tasks[$i]}" >"$OUTPUT_ROOT/logs/${tasks[$i]}.log" 2>&1 &
  client_pids+=("$!")
done
status=0
for i in "${!client_pids[@]}"; do
  if wait "${client_pids[$i]}"; then
    echo "[DONE] ${tasks[$i]}"
  else
    echo "[FAILED] ${tasks[$i]} (see $OUTPUT_ROOT/logs/${tasks[$i]}.log)" >&2
    status=1
  fi
done
exit "$status"
