#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-python}" -m torch.distributed.run \
  --nnodes="${NNODES:-2}" --nproc_per_node="${NPROC_PER_NODE:-16}" \
  --node_rank="${NODE_RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29719}" \
  -m training.dexjoco.train --config "${CONFIG:-training/configs/dexjoco_uni.yaml}" \
  --deepspeed training/configs/dexjoco_zero1.json "$@"
