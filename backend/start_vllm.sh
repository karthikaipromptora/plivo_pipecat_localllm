#!/usr/bin/env bash
set -euo pipefail

MODEL="${LOCAL_LLM_MODEL:-google/gemma-4-26B-A4B-it}"
PORT="${LOCAL_LLM_PORT:-8049}"

echo "Starting vLLM: model=$MODEL  port=$PORT"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --quantization fp8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --generation-config vllm \
    --enable-prefix-caching \
    --trust-remote-code \
    --no-enable-log-requests
