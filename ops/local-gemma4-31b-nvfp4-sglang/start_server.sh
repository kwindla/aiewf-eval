#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="aiewf-gemma4-31b-nvfp4"
MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
MODEL_REVISION="edafdf3dcaef23ff76f75b91edd6a4a975a399cf"
SGLANG_IMAGE="lmsysorg/sglang@sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2"
HF_CACHE_DIR="/home/khkramer/.cache/huggingface"
SGLANG_JIT_CACHE_DIR="/home/khkramer/.cache/sglang-tvm-ffi-gemma4-nvfp4"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists; stop it before starting a replacement." >&2
  exit 1
fi

mkdir -p "${SGLANG_JIT_CACHE_DIR}"

docker run --detach \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --volume "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  --volume "${SGLANG_JIT_CACHE_DIR}:/root/.cache/tvm-ffi" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "${SGLANG_IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --served-model-name google/gemma-4-31B-it \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16 \
    --quantization compressed-tensors \
    --fp4-gemm-backend cutlass \
    --kv-cache-dtype fp8_e4m3 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --context-length 32768 \
    --swa-full-tokens-ratio 1.0 \
    --max-running-requests 1 \
    --chunked-prefill-size 2048 \
    --max-prefill-tokens 2048 \
    --mem-fraction-static 0.90 \
    --sampling-defaults openai \
    --stream-interval 1 \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --enable-cache-report \
    --enable-metrics

echo "Started ${CONTAINER_NAME}; follow startup with:"
echo "  docker logs -f ${CONTAINER_NAME}"
