#!/usr/bin/env bash
set -euo pipefail

MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen3-14B}"
PORT="${LOCAL_LLM_PORT:-8049}"

# Force V0 engine — V1 engine in 0.19.x has a scheduler bug causing requests to hang
export VLLM_USE_V1=0

echo "Starting vLLM: model=$MODEL  port=$PORT"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --quantization fp8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --no-enable-log-requests
