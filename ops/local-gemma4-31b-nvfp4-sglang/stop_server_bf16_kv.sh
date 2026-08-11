#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="aiewf-gemma4-31b-nvfp4-bf16kv"

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker stop --timeout 30 "${CONTAINER_NAME}"
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker rm "${CONTAINER_NAME}"
fi
