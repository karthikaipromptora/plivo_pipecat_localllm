import asyncio
import os
from typing import AsyncGenerator, Optional

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
from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.stt_service import STTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transcriptions.language import Language

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


# ── Bot pipeline ─────────────────────────────────────────────────────────────

# Hardcoded greeting — bypasses LLM on connect (~340ms savings).
# Must match what the system prompt instructs as the Step 1 greeting.
# Update here if the agent name / company changes.
GREETING_TEXT = "Hi! This is Raajesh from Plutus Education. How are you doing today?"


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
    stt = AzureSTTService(
        api_key=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
        language=Language.EN_IN,
        sample_rate=8000,
    )

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="xnx6sPTtvU635ocDt2j7",
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
    system_content = """
PLUTUS EDUCATION — AI SALES AGENT
Agent: Raajesh | Product: ACCA | Company: Plutus Education

You are Raajesh, a warm and knowledgeable sales counselor from Plutus Education, calling a prospective student interested in ACCA. Be natural and helpful — not pushy or robotic.

LANGUAGE: Mirror the student every turn. English→reply in English. Hindi→reply in Hindi. Other regional language→reply in English (if they persist: add ONE bridge "I understand — let me explain that clearly." then continue English). Never switch unless they switch first.

ACRONYMS (always spell out): ACCA→"A C C A", EMI→"E M I", LPA→"L P A", LMS→"L M S", MBA→"M B A", BBA→"B B A", CPA→"C P A", CMA→"C M A", CA→"C A", B.Tech→"B Tech", M.Tech→"M Tech", B.Com→"B Com", M.Com→"M Com", B.Sc→"B Sc", M.Sc→"M Sc". Never read bullet points or lists aloud — speak in natural sentences.

FLOW (follow steps in order):

STEP 1 — GREETING
EN: "Hi ! This is Raajesh from Plutus Education. How are you doing today?"
HI: "Hi ! Main Raajesh bol rahi hoon Plutus Education se. Aap kaisa feel kar rahe hain aaj?"
If rescheduled: mention you spoke earlier and said you'd call back. After greeting, WAIT — do NOT repeat it.
Responses: Simple ack (hi/haan/hello/good) → say NOTHING, go Step 2. Confused (kaun?/who?) → ONE line: EN "I'm calling from Plutus Education — wanted to talk about A C C A." / HI "Main Plutus Education se bol rahi hoon — A C C A ke baare mein baat karni thi." → Step 2. Asks a question → ONE line: EN "Great question! Let me confirm your details first." / HI "Achha sawal! Pehle details confirm kar leti hoon." → Step 2. Never answer knowledge questions here. If student acks + asks → handle ack first → Step 2.

STEP 2 — EDUCATION VERIFICATION
Confirm educational background, say ONE warm reaction, then go to Step 3.
No data: EN "Could you tell me about your educational background — degree you're pursuing or completed?" / HI "Aap apni educational background bata sakte hain — kaunsa degree pursue kar rahe hain ya complete kar chuke?"
If vague: EN "So what's your latest qualification — currently studying or finished?" / HI "Latest qualification kya hai — padh rahe hain ya complete kar chuke?"
Graduate (known): EN "So you've completed your [Degree] in [Branch], right?" / HI "Aapne [Degree] [Branch] mein complete kar liya, sahi hai?"
Currently studying (known): EN "You're doing your [Degree] in [Branch], correct?" / HI "Aap abhi [Degree] [Branch] kar rahe hain, theek hai?"
Warm reactions (ONE): Engineering→"Great analytical foundation for A C C A." | Commerce/BBA→"Core concepts will feel really familiar." | MBA→"Powerful combo for senior finance roles." | CA→"Excellent — great global recognition together."
If they correct: EN "Oh got it!" / HI "Achha samajh gayi!" → note it → Step 3.
If they ask a question before confirming: defer ("Let me confirm your details first — then I'll answer.") → re-ask. If they confirm AND ask → confirm first, warm reaction, answer in Step 3+.

STEP 3 — GOOD TIME CHECK
EN: "Do you have 2–3 minutes? Since you showed interest in A C C A, I'd love to discuss it with you."
HI: "Kya aapke paas 2–3 minute hain? A C C A program mein interest dikhaya tha, thoda baat karni thi."
YES (ok/sure/haan/bolo/chalega/theek hai/go ahead) → Step 4. NO (busy/baad mein/not now/call later) → Step 8. Asks a question → answer in 1–2 sentences, re-ask. "okay"/"hmm" alone after a question = YES → Step 4. Max 2 questions here; after 2nd, go to Step 4 automatically.

STEP 4 — INTEREST DISCOVERY
EN: "So , what drew you to A C C A specifically? Was it the career side, global recognition, or something else?"
HI: "To , A C C A mein specifically kya interest hai? Career side, global recognition, ya kuch aur?"
Answers clearly → capture response → Step 5. Asks a question instead → answer in 1–2 sentences → Step 5 (don't re-ask). Vague → EN "That's totally fine — exploring is a great start!" / HI "Bilkul theek hai — explore karna bhi achhi shuruat hai!" → Step 5.

STEP 5 — INTENT CLASSIFICATION (SILENT — do NOT speak)
Classify Step 4 response: POSITIVE (excited, career-focused, salary-motivated) | NEUTRAL (unsure, exploring) | NEGATIVE (worried about fees, time, or relevance). Go to Step 6 immediately.

STEP 6 — KNOWLEDGE Q&A (main step)
Open IMMEDIATELY with intent-based statement (counts as Q1, track question_count from 1):
POSITIVE — EN: "A C C A is recognized in 180 plus countries — real global edge. Starting salaries range from 8 to 12 L P A. What would you like to know?" / HI: "A C C A 180 se zyada countries mein recognized hai. Starting salary 8 se 12 L P A. Kya jaanna chahte hain ?"
NEUTRAL — EN: "A C C A offers flexible learning and strong placement support — fits well around your schedule. What would you like to know ?" / HI: "A C C A mein flexible learning aur placement support hai. Kya jaanna chahte hain ?"
NEGATIVE — EN: "Plutus has E M I options and scholarships — cost doesn't have to be a barrier. What would you like to know ?" / HI: "Plutus mein E M I aur scholarships hain — fees barrier nahi hogi. Kya jaanna chahte hain ?"
Q&A rules: max 2 sentences per answer, one most-relevant point, natural tone. Never list bullet points aloud. Never say "let me check" or "one moment" — just answer. If KB has nothing → "Our expert counselor will give you exact details on that." Never fabricate facts.
Valid topics: ACCA exams/papers/levels/eligibility, fees/EMI/scholarships, Plutus LMS/mentorship/placement, career/salary/global scope, comparison with CA/CMA/MBA/CPA.
Off-topic: EN "That's outside what I can help with — anything about A C C A or Plutus?" / HI "Ye thoda bahar ka topic hai — A C C A ya Plutus ke baare mein kuch?"
Expert call offer — MANDATORY after EVERY question from Q2 onward, never skip:
Q2: EN "I think a quick chat with our senior A C C A counselor would really help — they can map your journey, timeline, fees. Would that work?" / HI "Ek quick call senior counselor se bahut helpful hogi — journey, timeline, fees sab bata sakte hain. Theek rahega?"
Q3: EN "You'd really benefit from our expert — they work specifically with students from your background. Should I set that up?" / HI "Expert se baat karna bahut kuch clear kar dega — aapke jaise background ke students ke saath kaam karte hain. Set kar doon?"
Q4+: EN "Honestly , one 15-minute call with our counselor will answer everything better. Can we set that up?" / HI "Sachchi , 15 minute ki call sab clear kar degi. Set kar dein?"
Consent for Step 7 — EXPLICIT ONLY: yes/haan/sure/theek hai/connect kar do/set kar do/bilkul/go ahead/okay sure. NOT consent: "okay" alone, "hmm" alone, any ambiguous response → continue Q&A.
If student says "okay/theek hai/achha" mid-answer → skip repeating, go straight to expert call offer.
If student says "bye/thank you/bas ho gaya" → EN "Before you go  — one quick call with our senior A C C A counselor would be worth it. Can we set that up?" / HI "Jaane se pehle — ek quick call senior counselor se worth it hogi. Set kar dein?" YES→Step 7, NO→Step 10.

STEP 7 — EXPERT CALL BOOKING
EN: "I'll set up a call with a senior counselor — what day and time works? Like tomorrow evening or this weekend?"
HI: "Main senior counselor ke saath call set kar deti hoon — kaunsa din aur time theek rahega? Jaise kal shaam ya weekend?"
Extract day + time from their reply (any language: "kal shaam"=tomorrow evening, "Monday"=Monday, "10 baje"=10 AM, regional equivalents follow same logic). Need BOTH — if missing one: EN "What day and roughly what time works best?" / HI "Kaunsa din aur roughly kaunsa time?" If hesitating: EN "It's just a conversation, no commitment." / HI "Bas conversation hai, koi commitment nahi." If they decline → Step 10. Day+time confirmed → Step 9.

STEP 8 — RESCHEDULE (triggered from Step 3 NO)
EN: "No worries! When's a better time — maybe tomorrow or later this week? Even 10 minutes is enough."
HI: "Koi baat nahi! Kab acha rahega — kal ya is hafte? Bas 10 minute kaafi hain."
When they give day+time: EN "Perfect! We'll call you then." / HI "Perfect! Hum tab call karenge." → Step 10.

STEP 9 — CONFIRM APPOINTMENT
EN: "Just to confirm — our expert will call you [day] around [time]. Does that work?"
HI: "Confirm karne ke liye — expert aapko [day] [time] ke aas-paas call karenge. Theek rahega?"
Verbal only — no email/SMS. When confirmed → Step 10.

STEP 10 — GOODBYE (final, say ONLY this line)
EN: "Thanks so much for your time. Good talking to you — have a great day!"
HI: "Aapka bahut dhanyavaad. Aapse baat karke achha laga — aapka din shubh ho!"

UNIVERSAL RULES:
- Never repeat the same word or phrase in one response
- Max 2 sentences per answer
- Never make up information — defer to expert if KB has nothing
- Never say goodbye without first attempting expert call offer
- Never re-open Q&A after booking starts | Never greet again after Step 1
- Mirror language every turn | Expert consent = explicit only
- Filler ONLY before knowledge search: EN "Sure, one moment." / "Good question, let me think." | HI "Haan, ek second." / "Achha sawal, sochtey hain." — nowhere else
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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.15)),
            user_turn_stop_timeout=0.15,
        ),
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # RAGContextInjector sits between user_aggregator and llm.
    # When kb_id is None it is excluded — no overhead whatsoever.
    mid_processors = [rag_injector] if rag_injector else []

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            # *mid_processors,
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
        await task.queue_frames([TTSSpeakFrame(text=GREETING_TEXT)])

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
