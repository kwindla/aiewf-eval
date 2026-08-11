#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="aiewf-gemma4-31b-nvfp4"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker stop --timeout 30 "${CONTAINER_NAME}"
  docker rm "${CONTAINER_NAME}"
else
  echo "Container ${CONTAINER_NAME} is not present."
fi
