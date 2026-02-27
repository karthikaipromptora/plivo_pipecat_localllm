#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMContextFrame, LLMRunFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.stt_service import STTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from deepgram import LiveOptions

load_dotenv(override=True)


# ── Lazy Deepgram (saves ~1.3s startup silence) ──────────────────────────────

class LazyDeepgramSTTService(DeepgramSTTService):
    """Connects to Deepgram in the background so the pipeline starts immediately.

    Without this, DeepgramSTTService.start() blocks for ~1.3s on the WebSocket
    handshake before the pipeline is ready. Since the bot speaks first, Deepgram
    has ~5-7s to connect before any user audio arrives — zero risk.
    """

    async def start(self, frame: StartFrame):
        await STTService.start(self, frame)
        self._settings["sample_rate"] = self.sample_rate
        asyncio.create_task(self._connect())

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if hasattr(self, "_connection"):
            try:
                if await self._connection.is_connected():
                    await self._connection.send(audio)
            except Exception:
                pass
        yield None


# ── RAG context injector ──────────────────────────────────────────────────────

class RAGContextInjector(FrameProcessor):
    """Injects semantically relevant KB chunks into LLM context before each turn.

    Sits between user_aggregator and llm in the pipeline. On every
    LLMContextFrame, it:
      1. Extracts the latest user message.
      2. Runs a GPU-accelerated vector search (~20ms).
      3. Injects the top matching chunks as a system message at position 1.
      4. Removes the previous injection so context doesn't grow unbounded.

    This avoids tool calling entirely. Works for any KB size — only the
    semantically relevant chunks for the current question are injected.
    Latency overhead: ~20ms per turn (vs 0 if no results found).
    """

    def __init__(self, kb_id: str):
        super().__init__()
        self._kb_id = kb_id
        self._rag_msg_idx: Optional[int] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._inject(frame.context)

        await self.push_frame(frame, direction)

    async def _inject(self, context: LLMContext):
        messages = context._messages

        # Find last non-seed user message
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content") != "begin":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    user_msg = content
                    break

        if not user_msg:
            return

        # Remove previous RAG injection to keep context clean
        if self._rag_msg_idx is not None:
            if self._rag_msg_idx < len(messages):
                messages.pop(self._rag_msg_idx)
            self._rag_msg_idx = None

        # Semantic search — ~20ms on GPU + local Qdrant
        from heplers.rag import rag_search
        result = await rag_search(query=user_msg, kb_id=self._kb_id)

        if result == "No relevant information found.":
            logger.debug("RAG: no relevant chunks found, skipping injection")
            return

        # Inject after system prompt (position 1)
        messages.insert(1, {
            "role": "system",
            "content": f"Relevant context from knowledge base:\n{result}",
        })
        self._rag_msg_idx = 1
        logger.info(f"RAG injected {len(result)} chars for query: {user_msg!r:.60}")


# ── Bot pipeline ─────────────────────────────────────────────────────────────

async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    kb_id: Optional[str] = None,
):
    """Run the voice bot pipeline.

    Args:
        transport:     Pipecat WebSocket transport.
        handle_sigint: Whether to handle SIGINT for clean shutdown.
        kb_id:         Qdrant knowledge base ID. When provided, a RAGContextInjector
                       is inserted between user_aggregator and llm. When None,
                       the injector is skipped entirely (zero overhead).
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
    )

    # ── STT ──────────────────────────────────────────────────────────────────
    stt = LazyDeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2-phonecall",
            endpointing=200,
        ),
    )

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="fPIfC3elMLbN9tNwMXkw",
        model="eleven_flash_v2_5",
    )

    # ── RAG injector (only inserted when kb_id is provided) ──────────────────
    #
    # Per-turn semantic injection — no tool calling required:
    #   User turn ends → LLMContextFrame flows downstream
    #   → RAGContextInjector embeds user message, searches Qdrant (~20ms)
    #   → Injects top-K chunks as system message at position 1
    #   → Groq LLM receives context with relevant KB content already baked in
    #   → Answers correctly with zero tool-call overhead
    #
    rag_injector = RAGContextInjector(kb_id=kb_id) if kb_id else None
    if kb_id:
        logger.info(f"RAG enabled for this call (kb_id={kb_id!r})")

    # ── System prompt ─────────────────────────────────────────────────────────
    system_content = (
        "You are a friendly conversational recruiter named Amit from S A S Consultants. "
        "Your very first message must greet the user and introduce yourself as Amit "
        "from S A S Consultants, then ask if it is a good time to talk. "
        "Do NOT repeat the introduction in later responses. "
        "If it is a good time to talk, mention you have a job opening for a Python "
        "Developer role and ask if they are interested. "
        "If they are interested, continue the conversation. "
        "If not interested, politely end the call. "
        "You should be confident, polite, and friendly. "
        "Your responses will be read aloud, so keep them concise and conversational. "
        "Avoid special characters or formatting. "
        "If your previous response appears cut off or incomplete in the conversation "
        "history, continue naturally without referencing it."
    )

    if kb_id:
        system_content += (
            " When a 'Relevant context from knowledge base' section appears in the "
            "conversation, use that information to answer the user's question accurately. "
            "NEVER invent job details not present in the provided context."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "begin"},
    ]

    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(
                stop_secs=0.4,       # was 0.15 — reduces false VAD triggers and double-interruptions
                start_secs=0.3,      # require 300ms of speech before triggering user turn start
                min_volume=0.7,      # ignore low-volume noise/sidetone from phone echo
            )),
            user_turn_stop_timeout=0.1,  # was 0.3 — more time before committing to end of turn
        ),
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # RAGContextInjector sits between user_aggregator and llm.
    # When kb_id is None it is excluded — no overhead whatsoever.
    mid_processors = [rag_injector] if rag_injector else []

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        *mid_processors,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Call started — RAG {'enabled' if kb_id else 'disabled'}")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


# ── Entry point ───────────────────────────────────────────────────────────────

async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Transport: {transport_type}")

    # Extract kb_id from body (passed via /start → answer_url → WebSocket → here)
    body = runner_args.body or {}
    kb_id: Optional[str] = body.get("kb_id") or None
    if kb_id:
        logger.info(f"kb_id received: {kb_id!r}")
    else:
        logger.info("No kb_id — RAG disabled for this call")

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, kb_id=kb_id)
