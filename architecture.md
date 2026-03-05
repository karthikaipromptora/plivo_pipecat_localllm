┌─────────────────────────────────────────────────────────────────┐
│                        INBOUND CALL FLOW                        │
└─────────────────────────────────────────────────────────────────┘

  POST /start
  (phone_number + kb_id?)
        │
        ▼
  ┌───────────┐    REST API     ┌─────────────┐
  │  FastAPI  │ ─────────────► │  Plivo API  │ ──► dials phone
  │ server.py │                └─────────────┘
  └───────────┘
        │
        │  Plivo calls GET/POST /answer
        ▼
  ┌───────────┐
  │  /answer  │ ◄── returns XML: <Stream mulaw 8kHz bidirectional>
  └───────────┘
        │
        │  Plivo opens WebSocket to /ws
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                        PIPECAT PIPELINE                          │
│                                                                  │
│  Plivo WS ──► STT ──► VAD + User Aggregator ──► LLM ──► TTS ──► Plivo WS  │
│  (mulaw)     Azure    Silero                   Groq  ElevenLabs (mulaw)    │
│  8kHz        EN_IN    stop=0.15s            llama-3.3  flash_v2_5  8kHz   │
│                                             -70b                          │
└──────────────────────────────────────────────────────────────────┘
        │
        │  on_client_connected → TTSSpeakFrame (hardcoded greeting)
        │  on_client_disconnected → task.cancel()


┌─────────────────────────────────────────────────────────────────┐
│                         RAG SUBSYSTEM                           │
│                       (⚠ not wired in)                          │
└─────────────────────────────────────────────────────────────────┘

  POST /upload (PDF)
        │
        ▼
  PyPDF extract ──► chunk (1000c / 200 overlap)
        │
        ▼
  sentence-transformers         Qdrant (Docker :6333)
  BAAI/bge-base-en-v1.5  ──►   collection: knowledge_base
  768 dims, GPU (RTX 4060)      filtered by kb_id


┌─────────────────────────────────────────────────────────────────┐
│                           STACK                                 │
└─────────────────────────────────────────────────────────────────┘

  Telephony   │  Plivo
  Framework   │  Pipecat
  STT         │  Azure Cognitive Speech  (EN_IN, 8kHz)
  LLM         │  Groq  →  llama-3.3-70b-versatile
  TTS         │  ElevenLabs  →  eleven_flash_v2_5
  VAD         │  Silero
  Embeddings  │  sentence-transformers  BAAI/bge-base-en-v1.5
  Vector DB   │  Qdrant  (local Docker)
  Server      │  FastAPI + uvicorn  :7860
  Tunnel      │  ngrok  (dev)
