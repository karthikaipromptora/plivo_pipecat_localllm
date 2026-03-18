#!/usr/bin/env bash
# Start vLLM with Qwen3-30B-A3B on NVIDIA L40S (44GB VRAM)
#
# Key flags:
#   --quantization fp8            — fits comfortably in ~16GB, preserves accuracy
#   --max-model-len 8192          — telephony context is short; reduces KV-cache memory
#   --gpu-memory-utilization 0.80 — leave headroom for Sarvam STT/TTS HTTP overhead
#   --default-chat-template-kwargs — disables chain-of-thought (thinking mode) so the
#                                   model answers immediately without <think>…</think>
#   --enable-prefix-caching       — caches the system prompt KV across turns (~30ms saved)
#   --disable-log-requests        — cleaner logs in production
#
# Fallback: replace model with Qwen/Qwen3-8B for lower VRAM usage / faster TTFT.

set -euo pipefail

MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen3-30B-A3B}"
PORT="${LOCAL_LLM_PORT:-8000}"

echo "Starting vLLM: model=$MODEL  port=$PORT"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --quantization fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching \
    --trust-remote-code
