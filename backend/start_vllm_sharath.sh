#!/usr/bin/env bash
set -euo pipefail

# Configuration
MODEL="${LOCAL_LLM_MODEL:-cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit}"
PORT="${LOCAL_LLM_PORT:-8049}"
SERVED_NAME="${SERVED_NAME:-google/gemma-4-26B-A4B-it}"

echo "Starting vLLM: Loading ${MODEL} as ${SERVED_NAME} on port ${PORT}"

exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_NAME}" \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.8 \
    --enable-prefix-caching \
    --trust-remote-code \
    --no-enable-log-requests \
    --generation-config vllm