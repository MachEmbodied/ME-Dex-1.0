#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${CHECKPOINT:?Set CHECKPOINT to the policy checkpoint directory}"
exec "${PYTHON:-python}" -m runtime.dexjoco.server \
  --config "${CONFIG:-training/configs/dexjoco_uni.yaml}" \
  --checkpoint "$CHECKPOINT" --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-21000}" --device "${DEVICE:-cuda:0}" "$@"
