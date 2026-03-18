import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMRunFrame,
    StartFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
# from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
# from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

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

        if (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
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
        messages.insert(
            1,
            {
                "role": "system",
                "content": f"Relevant context from knowledge base:\n{result}",
            },
        )
        self._rag_msg_idx = 1
        logger.info(f"RAG injected {len(result)} chars for query: {user_msg!r:.60}")


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.debug(f"STT: [{frame.text}]")
        await self.push_frame(frame, direction)


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    kb_id: Optional[str] = None,
    body: Optional[dict] = None,
):
    """Run the voice bot pipeline.

    Args:
        transport:     Pipecat WebSocket transport.
        handle_sigint: Whether to handle SIGINT for clean shutdown.
        kb_id:         Qdrant knowledge base ID. When provided, a RAGContextInjector
                       is inserted between user_aggregator and llm. When None,
                       the injector is skipped entirely (zero overhead).
    """

    llm = OpenAILLMService(
        api_key="local",
        base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1"),
        model=os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-30B-A3B"),
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        model="saaras:v2.5",
        sample_rate=8000,
        params=SarvamSTTService.InputParams(
            vad_signals=True,
            prompt="You can transcribe in only english, hindi or telugu with respect to wha user asks.",  # server-side VAD → UserStarted/StoppedSpeakingFrame
        ),
    )

    # stt = WhisperSTTService(
    #     model=Model.LARGE_V3_TURBO,
    #     device="cuda",
    #     compute_type="float16",
    # )

    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        model="bulbul:v3-beta",
        voice_id="priya",
        sample_rate=8000,
        params=SarvamTTSService.InputParams(
            language=Language.EN_IN,
            pace=1.2,
            temperature=0.7,
        ),
    )

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY", ""),
    #     voice_id="7c6219d2-e8d2-462c-89d8-7ecba7c75d65",
    #     model="sonic-3",
    #     sample_rate=8000,
    # )

    rag_injector = RAGContextInjector(kb_id=kb_id) if kb_id else None
    if kb_id:
        logger.info(f"RAG enabled for this call (kb_id={kb_id!r})")

    # Extract rider variables from body_data (sent by the UI via /start)
    b = body or {}
    rider_name    = b.get("rider_name",    "the rider")
    vehicle_model = b.get("vehicle_model", "your vehicle")
    end_date      = b.get("end_date",      "the end date")
    amount        = b.get("amount",        "the due amount")
    call_day      = b.get("call_day",      "T-2")
    language      = b.get("language",      "English")

    greeting_text = f"Hi I am calling from OptiMotion, is this {rider_name}?"

    system_content = f"""
       OPTIMOTION — RENEWAL REMINDER BOT
        Agent: OptiMotion Renewal Bot | Company: OptiMotion

        You are an automated renewal reminder calling on behalf of OptiMotion.

        CALL VARIABLES (injected at runtime):
        - Rider Name: {rider_name}
        - Vehicle Model: {vehicle_model}
        - End Date: {end_date}
        - Amount Due: rupees {amount}
        - Call Day: {call_day}
        - Language: {language}

        Speak in {language} throughout the call. If the rider switches language mid-call, mirror them immediately.

        NUMBERS/PHONE: Always read phone numbers digit by digit. Read "9121581421" as "9 1 2 1 5 8 1 4 2 1". Read amounts as natural speech: "Rs. 1500" as "rupees fifteen hundred".

        GREETING (always start here):
        "Hi, is this {rider_name}?"
        - Rider confirms → go to MAIN MESSAGE for the call day
        - Rider is not available / wrong number → "Sorry to disturb, thank you." → end call
        - No answer → leave voicemail: "Hi {rider_name}, OptiMotion here. Your {vehicle_model} plan ends on {end_date}. Please renew on WhatsApp or call 9 1 2 1 5 8 1 4 2 1. Thanks." → end call
        
        With respect to CALL VARIABLES Details, pick up the correct main message and continue the convetrsation.  

        MAIN MESSAGE — T-2 (2 days before end date):
        Tone: Casual, friendly. Lead with the Rs. 100 discount.
        "Hey {rider_name}, Your {vehicle_model} plan ends in 2 days — {end_date}. Rent is Rs. {amount}. If you pay today you get Rs. 100 off. Payment link and QR are on WhatsApp."

        MAIN MESSAGE — T-1 (1 day before end date):
        Tone: Firm but helpful. No discount anymore — focus on avoiding service break.
        "Hey {rider_name}, Your {vehicle_model} plan ends tomorrow — {end_date}. Rs. {amount} due. Please pay today so there is no break in service. QR is on WhatsApp."

        MAIN MESSAGE — T0 (end date is today):
        Tone: Direct, urgent. Vehicle can get locked.
        "Hey {rider_name}, Your {vehicle_model} plan ends today — {end_date}. Rs. {amount} due. Please pay now, vehicle can be locked by end of day if it is not done. QR is on WhatsApp."

        RESPONSE BRANCHES (handle whichever the rider says):

        Rider says okay / will pay now:
        → "Perfect. Use the QR on WhatsApp, it is easier. Any trouble, call 9 1 2 1 5 8 1 4 2 1."

        Rider says will pay later / tomorrow:
        T-2 → "Sure, no problem. Just so you know the Rs. 100 off is only if you pay today. We will call again tomorrow."
        T-1 → "Okay, just make sure it is before end of day. If it is not done by then the vehicle could get locked."
        T0 → "Please do it as soon as possible — vehicle could get locked by end of day. Call 9 1 2 1 5 8 1 4 2 1 if you need help."

        Rider says already paid:
        → "Oh great, thanks! Sometimes it takes a few hours to show up. If it still does not update, call 9 1 2 1 5 8 1 4 2 1."

        Rider wants to stop / does not want to continue:
        → "Got it. Plan runs till {end_date}. For returning the vehicle or anything else, call 9 1 2 1 5 8 1 4 2 1."

        Rider asks for more time (T0 only):
        → "I cannot do that from here. Please call 9 1 2 1 5 8 1 4 2 1 right now and they will sort it out before the vehicle gets locked."

        Rider says amount is wrong:
        → "Please call 9 1 2 1 5 8 1 4 2 1 — they can check and fix it right away."

        Any other question / anything the bot cannot handle:
        → "For that, please call 9 1 2 1 5 8 1 4 2 1 — they will help you out."

        ENDING THE CALL:
        Once the rider's concern is addressed (they said okay, redirected to support, or wants to stop) — say "Thanks, have a good day." and end.

        UNIVERSAL RULES:
        - Never use emojis, symbols, or special characters — plain text only
        - Max 2 sentences per response
        - Never make up information — redirect to 9 1 2 1 5 8 1 4 2 1 for anything you cannot confirm
        - Do not repeat the same phrase twice in one response
        - Mirror language every turn — do not switch back to English if rider is in Hindi or Telugu
        - Never write "Rs." — always say "rupees [amount]" e.g. "rupees 2 thousand" or "rupees 1 thousand 500"
        - Never write ordinal dates like "20th" — write "20 March" or "twentieth of March"
        - For vehicle models, spell acronyms without hyphens: EV-3 → "E V 3", not "E V-3"
        - Company name is always "OptiMotion" — never "Optimotion" or "optimotion"
        - Never split amounts across sentences — write the full amount in one phrase
        - Once the main message has been delivered and the rider has responded, do NOT repeat it again — the conversation has moved forward
        - Once goodbye has been exchanged, if the rider says anything else ("okay", "hmm", "hello"), just say "Have a good day." and stop — never restart the pitch

        """

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
            user_turn_stop_timeout=0.2,
        ),
    )

    mid_processors = [rag_injector] if rag_injector else []

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptionLogger(),
            user_aggregator,
            *mid_processors,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

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
        # Speak greeting directly via TTS — skips Groq entirely (~340ms saved).
        await task.queue_frames([TTSSpeakFrame(text=greeting_text)])

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

    await run_bot(transport, runner_args.handle_sigint, kb_id=kb_id, body=body)
