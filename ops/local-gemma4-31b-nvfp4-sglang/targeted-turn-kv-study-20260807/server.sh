#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
MODEL_REVISION="edafdf3dcaef23ff76f75b91edd6a4a975a399cf"
SGLANG_IMAGE="lmsysorg/sglang@sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2"
HF_CACHE_DIR="/home/khkramer/.cache/huggingface"
SGLANG_JIT_CACHE_DIR="/home/khkramer/.cache/sglang-tvm-ffi-gemma4-nvfp4"

usage() {
  echo "usage: $0 start|wait|stop|status|logs --kv fp8|bf16 [--geometry compact|historical] [--sampling seeded|native|batch-invariant] [--stage NAME] [--port PORT]" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
action=$1
shift
kv=""
geometry="compact"
sampling="seeded"
stage="adhoc"
port="30000"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kv) kv=$2; shift 2 ;;
    --geometry) geometry=$2; shift 2 ;;
    --sampling) sampling=$2; shift 2 ;;
    --stage) stage=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$kv" == "fp8" || "$kv" == "bf16" ]] || usage
[[ "$geometry" == "compact" || "$geometry" == "historical" ]] || usage
[[ "$sampling" == "seeded" || "$sampling" == "native" || "$sampling" == "batch-invariant" ]] || usage
[[ "$stage" =~ ^[A-Za-z0-9._-]+$ ]] || usage
if [[ "$geometry" == "historical" && "$kv" != "fp8" ]]; then
  echo "historical geometry is defined only for the FP8 bridge arm" >&2
  exit 2
fi

container="aiewf-gemma-kv-target-${kv}-${geometry}-${sampling}"
health="http://127.0.0.1:${port}/health"
log_dir="${HERE}/server-logs"
mkdir -p "$log_dir" "$SGLANG_JIT_CACHE_DIR"

capture_container_artifacts() {
  local phase=$1
  local timestamp
  local instance_id
  local prefix
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  instance_id=$(docker inspect --format '{{.Id}}' "$container")
  prefix="${log_dir}/${container}.${stage}.${timestamp}.${instance_id:0:12}.${phase}"
  docker logs "$container" >"${prefix}.log" 2>&1 || true
  docker inspect "$container" >"${prefix}.inspect.json"
  echo "provenance log: ${prefix}.log"
  echo "provenance inspect: ${prefix}.inspect.json"
}

case "$action" in
  start)
    if docker ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
      echo "Container $container already exists; stop it before replacing it." >&2
      exit 1
    fi
    dtype="bfloat16"
    [[ "$kv" == "fp8" ]] && dtype="fp8_e4m3"
    geometry_args=(--max-total-tokens 16000 --swa-full-tokens-ratio 0.35)
    if [[ "$geometry" == "historical" ]]; then
      geometry_args=(--swa-full-tokens-ratio 1.0)
    fi
    sampling_args=(--attention-backend triton)
    sampling_env=()
    sampling_mount=()
    if [[ "$sampling" == "seeded" ]]; then
      sampling_args+=(--sampling-backend pytorch)
      sampling_env=(--env SGLANG_HONOR_REQUEST_SEED_WITHOUT_BATCH_INVARIANCE=1)
      sampling_mount=(--volume "${HERE}/shims:/study-shims:ro" --env PYTHONPATH=/study-shims)
    elif [[ "$sampling" == "batch-invariant" ]]; then
      sampling_args+=(--enable-deterministic-inference)
    fi
    docker run --detach \
      --name "$container" \
      --gpus all \
      --network host \
      --ipc host \
      --shm-size 32g \
      --volume "${HF_CACHE_DIR}:/root/.cache/huggingface" \
      --volume "${SGLANG_JIT_CACHE_DIR}:/root/.cache/tvm-ffi" \
      --env HF_HUB_OFFLINE=1 \
      --env TRANSFORMERS_OFFLINE=1 \
      "${sampling_env[@]}" \
      "${sampling_mount[@]}" \
      "$SGLANG_IMAGE" \
      python3 -m sglang.launch_server \
        --model-path "$MODEL_ID" \
        --revision "$MODEL_REVISION" \
        --served-model-name google/gemma-4-31B-it \
        --host 0.0.0.0 \
        --port "$port" \
        --dtype bfloat16 \
        --quantization compressed-tensors \
        --fp4-gemm-backend cutlass \
        --kv-cache-dtype "$dtype" \
        "${sampling_args[@]}" \
        --disable-cuda-graph \
        --skip-server-warmup \
        --context-length 32768 \
        "${geometry_args[@]}" \
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
    echo "started $container at http://127.0.0.1:${port}"
    ;;
  wait)
    deadline=$((SECONDS + 1200))
    until curl --fail --silent --show-error "$health" >/dev/null 2>&1; do
      if ! docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null | grep -Fxq true; then
        docker logs --tail 200 "$container" 2>&1 || true
        exit 1
      fi
      if (( SECONDS >= deadline )); then
        echo "server did not become healthy within 20 minutes" >&2
        exit 1
      fi
      sleep 5
    done
    capture_container_artifacts ready
    echo "$container ready"
    ;;
  stop)
    if docker ps -a --format '{{.Names}}' | grep -Fxq "$container"; then
      capture_container_artifacts pre-stop
      docker stop --time 30 "$container" >/dev/null
      capture_container_artifacts stopped
      docker rm "$container" >/dev/null
      echo "stopped and removed $container"
    else
      echo "$container is absent"
    fi
    ;;
  status)
    docker ps -a --filter "name=^/${container}$" --format '{{.Names}}\t{{.Status}}'
    ;;
  logs)
    docker logs --tail 300 "$container"
    ;;
  *) usage ;;
esac
