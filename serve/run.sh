#!/usr/bin/env bash
# Serve an EXL3 checkpoint (uniform or mixed K=2..7) with the patched exl3xpu image.
# This is the serve command used for the checks in b70/README.md.
#
#   MODEL_DIR  (required) local EXL3 model directory, mounted read-only
#   IMAGE      image tag built from b70/serve/Dockerfile (default exl3xpu-b70:mixedk)
#   GPU_INDEX  card index passed to serve.py --gpu (default 0)
#   PORT       OpenAI-compatible HTTP port (default 8022)
#   NAME       container name (default exl3xpu-b70)
# Extra arguments are appended to serve.py, e.g. --set vllm.speculative_config=null
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR to the EXL3 model directory}"
IMAGE=${IMAGE:-exl3xpu-b70:mixedk}
GPU_INDEX=${GPU_INDEX:-0}
PORT=${PORT:-8022}
NAME=${NAME:-exl3xpu-b70}
RENDER_GID=$(stat -c %g /dev/dri/renderD128)

exec docker run -d --name "$NAME" \
  --device=/dev/dri --group-add "$RENDER_GID" \
  --network=host --memory=30g \
  -v "$MODEL_DIR:/model:ro" \
  --entrypoint python3 "$IMAGE" \
  /opt/exl3xpu/scripts/serve.py /opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw \
  --gpu "$GPU_INDEX" --model-path /model --port "$PORT" \
  --set vllm.gpu_memory_utilization=0.90 --set vllm.max_model_len=65536 \
  "$@"
