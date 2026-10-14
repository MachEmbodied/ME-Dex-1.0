#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 TASK SERVER_URL OUTPUT" >&2
  exit 64
fi
TASK="$1"; SERVER_URL="$2"; OUTPUT="$3"
: "${DEXJOCO_ROOT:?Set DEXJOCO_ROOT to the DexJoCo simulator repository}"
DEXJOCO_PY="${DEXJOCO_PY:-python}"
export PYTHONPATH="$ROOT:$DEXJOCO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$ROOT/runtime/dexjoco/configs/$TASK.yaml" ]] || { echo "Task config missing: $TASK" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT" >&2; exit 1; }
CLEAN_LD=""
IFS=: read -ra entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${entries[@]}"; do
  case "$entry" in /opt/ac2/*|/usr/local/PPU_SDK*) continue ;; esac
  CLEAN_LD="${CLEAN_LD:+$CLEAN_LD:}$entry"
done
GL_BACKEND=""
for gl in "${MUJOCO_GL:-egl}" osmesa; do
  if env -u LD_PRELOAD LD_LIBRARY_PATH="$CLEAN_LD" MUJOCO_GL="$gl" PYOPENGL_PLATFORM="$gl" \
    "$DEXJOCO_PY" -c 'from mujoco import GLContext; c=GLContext(8,8); c.make_current(); c.free()' >/dev/null 2>&1; then
    GL_BACKEND="$gl"; break
  fi
done
[[ -n "$GL_BACKEND" ]] || { echo "MuJoCo GLContext failed for EGL and OSMesa" >&2; exit 1; }
VIDEO_ARGS=(--record-video)
[[ "${RECORD_VIDEO:-1}" != 0 ]] || VIDEO_ARGS=(--no-record-video)
exec env -u LD_PRELOAD LD_LIBRARY_PATH="$CLEAN_LD" MUJOCO_GL="$GL_BACKEND" PYOPENGL_PLATFORM="$GL_BACKEND" \
  "$DEXJOCO_PY" -m runtime.dexjoco.evaluate \
  --config "$ROOT/runtime/dexjoco/configs/$TASK.yaml" --task "$TASK" \
  --server-url "$SERVER_URL" --output "$OUTPUT" --episodes "${EPISODES:-50}" \
  --seed "${SEED:-0}" --max-env-steps "${MAX_ENV_STEPS:-1200}" \
  --chunk-steps 16 --sparse-densify --smooth-alpha 1.0 \
  --episode-seed-mode legacy-stream --no-rand-full --no-randomize-dynamics "${VIDEO_ARGS[@]}"
