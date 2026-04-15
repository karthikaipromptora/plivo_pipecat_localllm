import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    EndFrame,
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
# from pipecat.serializers.vobiz import VobizFrameSerializer
# from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.services.sarvam.stt import SarvamSTTService

from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from pipecat.services.sarvam.tts import SarvamTTSService

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
            logger.debug(f"STT: [{frame.text}] | lang: {frame.language}")
        await self.push_frame(frame, direction)





async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    kb_id: Optional[str] = None,
    body: Optional[dict] = None,
    transcript_out: Optional[list] = None,
):
    """Run the voice bot pipeline.

    Args:
        transport:     Pipecat WebSocket transport.
        handle_sigint: Whether to handle SIGINT for clean shutdown.
        kb_id:         Qdrant knowledge base ID. When provided, a RAGContextInjector
                       is inserted between user_aggregator and llm. When None,
                       the injector is skipped entirely (zero overhead).
    """

    # llm = OpenAILLMService(
    #     api_key="local",
    #     base_url=os.getenv("LOCAL_LLM_URL", "http://164.52.198.104:8049/v1"),
    #     model=os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-14B"),
    # )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model="gpt-4-o-mini",
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        model="saaras:v3",
        sample_rate=8000,
        params=SarvamSTTService.InputParams(
            vad_signals=True,
            mode="transcribe",  # transcribe in original language — no translation to English
        ),
    )

    # stt = WhisperSTTService(
    #     model=Model.LARGE_V3_TURBO,
    #     device="cuda",
    #     compute_type="float16",
    # )

    # Extract rider variables from body_data (sent by the UI via /start)
    from num2words import num2words

    b = body or {}
    rider_name    = b.get("rider_name",    "the rider")
    vehicle_model = b.get("vehicle_model", "your vehicle")
    end_date      = b.get("end_date",      "the end date")
    call_day      = b.get("call_day",      "T-2")
    language      = b.get("language",      "English")

    # Convert amount to English cardinal words so LLM never sees a numeral
    _amount_raw = str(b.get("amount", "")).strip()
    try:
        amount = num2words(int(_amount_raw), lang="en")   # e.g. "two thousand"
    except (ValueError, TypeError):
        amount = _amount_raw or "the due amount"

    # Convert end_date to spoken English words so LLM never translates it
    # Input: "20 March 2025" or "2025-03-20" → "twenty March twenty twenty five"
    def _date_to_words(raw: str) -> str:
        from datetime import datetime
        raw = raw.strip()
        if not raw or raw == "the end date":
            return raw
        dt = None
        for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return raw  # fallback: return as-is
        day_word   = num2words(dt.day, lang="en")           # "twenty"
        month_word = dt.strftime("%B")                      # "March"
        year       = dt.year                                # 2025
        # Split year into two halves: 2025 → "twenty" + "twenty five"
        century, decade = divmod(year, 100)
        if decade == 0:
            year_word = num2words(century, lang="en") + " hundred"  # "twenty hundred"
        else:
            year_word = num2words(century, lang="en") + " " + num2words(decade, lang="en")  # "twenty twenty five"
        return f"{day_word} {month_word} {year_word}"

    end_date_words = _date_to_words(str(end_date))

    agent_name   = "Rajesh"
    agent_gender = "male"

    _lang_code_map = {
        "English": "en-IN",
        "Hindi":   "hi-IN",
        "Telugu":  "te-IN",
    }
    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        target_language_code=_lang_code_map.get(language, "en-IN"),
        model="bulbul:v3",
        speaker="ratan",
    )
    logger.info(f"TTS: Sarvam bulbul:v3 ratan ({language} → {_lang_code_map.get(language, 'en-IN')}, {call_day})")

    rag_injector = RAGContextInjector(kb_id=kb_id) if kb_id else None
    if kb_id:
        logger.info(f"RAG enabled for this call (kb_id={kb_id!r})")

    hindi_verb    = "बोल रही हूँ" if agent_gender == "female" else "बोल रहा हूँ"
    greetings = {
        "English": f"Hi, I am {agent_name} from OptiMotion, is this {rider_name}?",
        "Hindi":   f"Hi, मैं {agent_name} {hindi_verb} OptiMotion से। क्या आप {rider_name} हैं?",
        "Telugu":  f"Hi, నేను {agent_name}, OptiMotion నుండి। మీరు {rider_name} గారా?",
    }
    greeting_text = greetings.get(language, greetings["English"])

    # ── Shared blocks (embedded in each per-day template) ─────────────────────

    _common_context = f"""
        CALL CONTEXT:
        - Rider: {rider_name}
        - Vehicle: {vehicle_model}
        - Plan End Date: {end_date_words}  ← say EXACTLY "{end_date_words}" — do NOT translate to Hindi or Telugu
        - Amount Due: {amount} rupees  ← say EXACTLY "{amount} rupees" — do NOT translate "{amount}" to any other language

        PAYMENT INFO:
        - Amount due: {amount} rupees
        - How to pay: QR code and payment link sent on WhatsApp
    """

    _common_conversation_rules = f"""
        HOW TO HANDLE THE CONVERSATION:

        After greeting is done, tell the user that their vechile plan status with respect to plan end date {end_date_words}, and tell them that the about they should pay to continue with the plan is {amount}, tell them that the payment link is already sent via whatsapp and ask the user to pay via link.
        continue the conversation answer user queries. Any question you dont have information, tell them that you are going to raise this issue with the support team and support team will get back to the customer really soon. Never repeat the sentence that user has already said. Always respond in a warm, human-like manner
    """

    _language_block = f"""
        LANGUAGE — This call starts in {language}. Begin in {language}.
        Always respond in the SAME language the user is currently speaking. Detect it from their message and match it immediately. If they switch language mid-call, switch with them in the very next response.

        NUMBERS RULE — applies in every language, never break this:
        - Amount: always say "{amount} rupees" — NEVER translate "{amount}" into Hindi or Telugu words
        - Date: always say "{end_date_words}" — NEVER translate into Hindi or Telugu words

        ── English rules ──
        Style: Plain, warm, conversational — not scripted.
        CORRECT: "Your payment of {amount} rupees is due today."

        ── Hindi rules ──
        Style: Warm, casual Hinglish — mix English words naturally into every response. Never pure Hindi.
        Script: Devanagari only for Hindi words — CRITICAL: no Roman transliteration ever, degrades TTS quality.
        Gendered grammar: agent is {agent_gender} → use {"बोल रही हूँ" if agent_gender == "female" else "बोल रहा हूँ"}.
        Sentence endings: every Hindi sentence must end with । (Devanagari danda), NEVER a period (.)
        Sentence length: keep each sentence under 20 words — long sentences cause unnatural TTS breathing.
        Line breaks: use \\n between sentences in multi-sentence responses.
        Avoid Sanskrit-heavy words — use simple colloquial Hindi (खत्म not समाप्त, problem not समस्या).
        MANDATORY substitutions — always use the English word, never the Hindi equivalent:
          भुगतान → pay | भुगतान करें → pay करें | कृपया → please
          वाहन → vehicle | सदस्यता → subscription | समस्या → problem
          धन्यवाद → thank you | कॉलबैक → callback | सहायता → support
          लॉक → lock | अनलॉक → unlock | लिंक → link
        CORRECT: "please WhatsApp पर भेजे गए link से {amount} rupees pay करें।"
        CORRECT: "मैं यह issue support team को raise करता हूँ, वो आपको call back करेंगे।"
        CORRECT: "आपका vehicle अभी lock है, {amount} rupees pay करने के बाद unlock होगा।"
        WRONG:   "कृपया दो हज़ार रुपये का भुगतान करें।"

        ── Telugu rules ──
        Style: Casual Indian Telugu mixed with English words naturally. Never formal or pure Telugu.Also make sure you are giving sentences in a proper format without any grammatical errors.Also make sure to give meaningful sentences which are easy to understand for the customers.
        Script: Telugu script only for all Telugu words — CRITICAL: no Roman transliteration ever, degrades TTS quality.
        Sentence endings: every Telugu sentence must end with । (danda), NEVER a period (.)
        Sentence length: keep each sentence under 20 words — long sentences cause unnatural TTS breathing.
        Line breaks: use \\n between sentences in multi-sentence responses.
        MANDATORY substitutions — always use the English word, never the Telugu equivalent:
          వాహనం → vehicle | చెల్లింపు → pay | చెల్లించండి → pay చేయండి
          దయచేసి → please | ధన్యవాదాలు → thank you | సేవ → service
          ప్రణాళిక → plan | సమాచారం → information | సభ్యత్వం → subscription
          సమస్య → problem | మద్దతు → support | లాక్ → lock | అన్‌లాక్ → unlock
        CORRECT: "please WhatsApp లో పంపిన link ద్వారా {amount} rupees pay చేయండి।"
        CORRECT: "నేను ఈ issue support team కి raise చేస్తాను, వాళ్ళు మీకు call back చేస్తారు।"
        CORRECT: "మీ vehicle ఇప్పుడు lock అయింది, {amount} rupees pay చేసిన తర్వాత unlock అవుతుంది।"
        WRONG:   "దయచేసి రెండు వేల రూపాయలు చెల్లించండి."
    """

    _format_rules = """
        FORMATTING RULES:
        - Plain text only — no emojis, no symbols, no markdown
        - Max 2 sentences per response — this is a voice call, keep it short
        - Never write "Rs." — always say "rupees" in words
        - Vehicle model without hyphens: EV-3 → E V 3
        - Company name is always "OptiMotion"
        - Hindi and Telugu sentences must end with । (danda), never a period
        - Keep every sentence under 20 words — Sarvam TTS breathes unnaturally on long sentences
        - Never use ellipsis (...) — causes choppy robotic delivery in Sarvam TTS
        - Use \\n between sentences in multi-sentence responses
        - No Roman transliteration of any Hindi or Telugu words — CRITICAL for TTS quality
    """

    # ── Per-day templates ──────────────────────────────────────────────────────

    if call_day == "T0":
        system_content = f"""
        You are {agent_name}, a {agent_gender} collection agent calling on behalf of OptiMotion.
        You are calling {rider_name} about their {vehicle_model} subscription plan — payment is due TODAY.
        You have very good understanding in all three languages: English, Hindi and Telugu.You can Understand and speak in all three languages fluently. Make sure you give proper responses with proper sentence formations in all three languages. The call can start in any of the three languages, so be prepared to switch between them as needed.

        YOUR SITUATION:
        Today is the payment due date. The rider has not paid yet.
        Be direct and warm — sound like a real human, not a robot. Slightly impatient but polite.
        Do NOT mention vehicle locking in this call.

        YOUR GOAL:
        State the due amount and WhatsApp payment link ONCE. Then listen and respond to what the rider says.
        Do NOT loop back or repeat — one mention, one response, close the call.
        {_common_context}
        {_common_conversation_rules}
        {_language_block}
        {_format_rules}
        """

    elif call_day == "T-1":
        system_content = f"""
        You are {agent_name}, a {agent_gender} collection agent calling on behalf of OptiMotion.
        You are calling {rider_name} about their {vehicle_model} subscription — payment was due YESTERDAY and has not been received.
        You have very good understanding in all three languages: English, Hindi and Telugu.You can Understand and speak in all three languages fluently. Make sure you give proper responses with proper sentence formations in all three languages. The call can start in any of the three languages, so be prepared to switch between them as needed.

        YOUR SITUATION:
        Payment is 1 day overdue the subscription day was {end_date_words}, today its 1 day after subscription date. If the rider does not pay TODAY, their vehicle will be locked tomorrow.
        Be firm and direct. Slightly annoyed but still professional. Sound like a real human — not a call-center robot.
        A touch of dry humour is okay — but keep it brief.
        English: "better late than never, right?" | Hindi: "late है लेकिन okay है, right?" | Telugu: "late అయింది కానీ okay, right?"

        YOUR GOAL:
        Make the situation clear ONCE — pay today or vehicle locks tomorrow. Then respond to what the rider says and close.
        Do NOT keep restating the deadline after you've said it once.
        {_common_context}
        {_common_conversation_rules}
        {_language_block}
        {_format_rules}
        """

    else:  # T-2
        system_content = f"""
        You are {agent_name}, a {agent_gender} collection agent calling on behalf of OptiMotion.
        You are calling {rider_name} about their {vehicle_model} subscription — payment is 2 days overdue and the vehicle is NOW LOCKED.
        You have very good understanding in all three languages: English, Hindi and Telugu.You can Understand and speak in all three languages fluently. Make sure you give proper responses with proper sentence formations in all three languages. The call can start in any of the three languages, so be prepared to switch between them as needed.

        YOUR SITUATION:
        The vehicle is locked. The rider needs to pay {amount} rupees immediately to unlock it. The subscription date was {end_date_words}, today its 2 day after subscription date.
        Be very firm and urgent. You are clearly not happy. Sound like a real, slightly frustrated human.
        Dry humour is okay — but do NOT soften the message.
        English: "I'm sure you enjoy walking, but let's fix this." | Hindi: "walking enjoy कर रहे हो क्या, chalo fix करते हैं।" | Telugu: "walking enjoy చేస్తున్నారా, let's fix this।"
        The vehicle IS locked. State it clearly, ONCE.

        YOUR GOAL:
        Inform them the vehicle is locked, state the amount and WhatsApp link ONCE, then respond to what they say and close.
        Do NOT repeat the lock warning or the amount after you've said it once.
        {_common_context}
        {_common_conversation_rules}
        {_language_block}
        {_format_rules}
        """

    if kb_id:
        system_content += (
            " When a 'Relevant context from knowledge base' section appears in the "
            "conversation, use that information to answer the user's question accurately. "
            "NEVER invent job details not present in the provided context."
            "You have ability to give proper responses with proper sentence formations in Telugu, Hindi, English languages."
        )

    logger.info(f"SYSTEM PROMPT ({'─'*60})\n{system_content}\n{'─'*70}")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "begin"},
        {"role": "assistant", "content": greeting_text},
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

    @transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, websocket):
        logger.info("Call timeout at 55s — wrapping up")
        farewells = {
            "English": "I need to wrap up now. Thank you for your time, have a great day!",
            "Hindi":   "मुझे अब call end करनी है। आपके time के लिए thank you!",
            "Telugu":  "నేను ఇప్పుడు call end చేయాలి। మీ time కి thank you!",
        }
        farewell_text = farewells.get(language, farewells["English"])
        await task.queue_frame(TTSSpeakFrame(text=farewell_text))
        await asyncio.sleep(7)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=handle_sigint)
    try:
        await runner.run(task)
    finally:
        # Snapshot transcript regardless of how the call ended (timeout, hangup, error).
        # Runs after pipeline stops — zero latency impact on the live call.
        if transcript_out is not None and not transcript_out:
            for msg in context._messages[2:]:
                role    = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    transcript_out.append({"role": role, "text": content})
            logger.info(f"Transcript snapshot: {len(transcript_out)} turns")


# ── Entry point ───────────────────────────────────────────────────────────────


async def bot(runner_args: RunnerArguments, transcript_out: Optional[list] = None):
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
        params=PlivoFrameSerializer.InputParams(
            plivo_sample_rate=8000,
            auto_hang_up=True,
        ),
    )
    # serializer = VobizFrameSerializer(
    #     stream_id=call_data["stream_id"],
    #     call_id=call_data["call_id"],
    #     auth_id=os.getenv("VOBIZ_AUTH_ID", ""),
    #     auth_token=os.getenv("VOBIZ_AUTH_TOKEN", ""),
    #     params=VobizFrameSerializer.InputParams(
    #         vobiz_sample_rate=8000,
    #         auto_hang_up=True,
    #     ),
    # )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            session_timeout=55,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, kb_id=kb_id, body=body, transcript_out=transcript_out)
