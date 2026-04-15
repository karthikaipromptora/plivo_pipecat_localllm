#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""server.py

Webhook server to handle outbound call requests, initiate calls via VoBiz API,
and handle subsequent WebSocket connections for Media Streams.
"""

import base64
import json
import os
import sys
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from routers.upload import router as upload_router
from routers.auth import router as auth_router, get_current_user
import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Logging — show only what matters ─────────────────────────────────────────
from loguru import logger as _logger

_logger.remove()

def _log_filter(record):
    level = record["level"].no   # DEBUG=10 INFO=20 WARNING=30 ERROR=40
    name  = record["name"]
    msg   = record["message"]

    if level >= 30:                                          # always show WARNING+
        return True
    if name.startswith(("heplers", "routers", "__main__")): # our code: INFO+
        return level >= 20
    if "TTFB" in msg:                                        # STT / LLM / TTS timing
        return True
    if "transcript=" in msg:                                 # Sarvam live transcription
        return True
    if "Transcription: [" in msg:                            # Whisper transcription
        return True
    if "STT: [" in msg:                                      # clean transcript line
        return True
    if "Generating TTS" in msg:                              # bot response text
        return True
    return False

_logger.add(
    sys.stderr,
    filter=_log_filter,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)
# ───────────────────────────────────────────────────────────────────────────


load_dotenv(override=True)

# Path to the frontend UI
_FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")


# ----------------- HELPERS ----------------- #


async def _analyse_transcript(turns: list[dict]) -> dict:
    """Call gpt-4o-mini post-call to get sentiment + takeaway + callback."""
    import json
    from openai import AsyncOpenAI

    transcript_text = "\n".join(
        f"{t['role'].upper()}: {t['text']}" for t in turns
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a call analysis assistant. Analyze the given call transcript "
                    "and return ONLY a JSON object with exactly these three fields:\n"
                    "  - sentiment: one of 'positive', 'negative', or 'neutral'\n"
                    "  - takeaway: a 5-8 word phrase summarizing what happened on the call\n"
                    "  - callback: 'yes' or 'no' — whether the customer needs a follow-up call from support\n\n"
                    "Rules:\n"
                    "- sentiment is positive if the rider agreed to pay, already paid, or was cooperative\n"
                    "- sentiment is negative if the rider refused, was rude, or disconnected without engaging\n"
                    "- sentiment is neutral if the call was inconclusive or the rider had questions\n"
                    "- takeaway must be 5-8 words max, plain English, no punctuation\n"
                    "- callback is 'yes' if the rider had unresolved questions, complaints, disputes, or requested help that the agent could not provide\n"
                    "- callback is 'no' if the call ended cleanly — rider acknowledged, agreed to pay, already paid, or simply hung up\n"
                    "- Return ONLY the JSON object, nothing else\n\n"
                    'Example: {"sentiment": "positive", "takeaway": "rider agreed to pay by tomorrow", "callback": "no"}'
                ),
            },
            {
                "role": "user",
                "content": f"Analyze this call transcript:\n\n{transcript_text}",
            },
        ],
    )

    result = json.loads(response.choices[0].message.content or "{}")

    sentiment = result.get("sentiment", "neutral")
    if sentiment not in ("positive", "negative", "neutral"):
        sentiment = "neutral"
    takeaway = result.get("takeaway", "")
    callback = result.get("callback", "no")
    if callback not in ("yes", "no"):
        callback = "no"

    return {"sentiment": sentiment, "takeaway": takeaway, "callback": callback}


# async def make_plivo_call(
#     session: aiohttp.ClientSession, to_number: str, from_number: str, answer_url: str
# ):
#     """Make an outbound call using Plivo's REST API."""
#     auth_id = os.getenv("PLIVO_AUTH_ID")
#     auth_token = os.getenv("PLIVO_AUTH_TOKEN")
#     if not auth_id:
#         raise ValueError("Missing Plivo Auth ID (PLIVO_AUTH_ID)")
#     if not auth_token:
#         raise ValueError("Missing Plivo Auth Token (PLIVO_AUTH_TOKEN)")
#     headers = {"Content-Type": "application/json"}
#     data = {"to": to_number, "from": from_number, "answer_url": answer_url, "answer_method": "POST"}
#     url = f"https://api.plivo.com/v1/Account/{auth_id}/Call/"
#     auth = aiohttp.BasicAuth(auth_id, auth_token)
#     async with session.post(url, headers=headers, json=data, auth=auth) as response:
#         if response.status != 201:
#             error_text = await response.text()
#             raise Exception(f"Plivo API error ({response.status}): {error_text}")
#         result = await response.json()
#         return result


async def make_vobiz_call(
    session: aiohttp.ClientSession, to_number: str, from_number: str | None, answer_url: str
):
    """Make an outbound call using VoBiz's REST API."""
    auth_id = os.getenv("VOBIZ_AUTH_ID")
    auth_token = os.getenv("VOBIZ_AUTH_TOKEN")

    if not auth_id:
        raise ValueError("Missing VoBiz Auth ID (VOBIZ_AUTH_ID)")
    if not auth_token:
        raise ValueError("Missing VoBiz Auth Token (VOBIZ_AUTH_TOKEN)")

    headers = {
        "Content-Type": "application/json",
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
    }

    data = {
        "to": to_number,
        "from": from_number,
        "answer_url": answer_url,
        "answer_method": "POST",
    }

    url = f"https://api.vobiz.ai/api/v1/Account/{auth_id}/Call/"

    async with session.post(url, headers=headers, json=data) as response:
        if response.status != 201:
            error_text = await response.text()
            raise Exception(f"VoBiz API error ({response.status}): {error_text}")

        result = await response.json()
        return result


def get_websocket_url(host: str):
    """Construct WebSocket URL based on environment variables."""
    env = os.getenv("ENV", "local").lower()

    if env == "production":
        print("If deployed in a region other than us-west (default), update websocket url!")
        ws_url = "wss://api.pipecat.daily.co/ws/plivo"
        # uncomment appropriate region url:
        # ws_url = wss://us-east.api.pipecat.daily.co/ws/plivo
        # ws_url = wss://eu-central.api.pipecat.daily.co/ws/plivo
        # ws_url = wss://ap-south.api.pipecat.daily.co/ws/plivo
        return ws_url
    else:
        return f"wss://{host}/ws"


# ----------------- API ----------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the embedding model at startup so the first RAG search
    # during a live call doesn't hit the model-load penalty.
    from heplers.rag import _embedding_model
    _embedding_model()  # loads SentenceTransformer + moves to GPU

    # Connect to Neon PostgreSQL and ensure tables exist
    from heplers.db import init_db, close_db
    await init_db()

    # Create aiohttp session for VoBiz API calls
    app.state.session = aiohttp.ClientSession()
    yield
    # Close session and DB pool on shutdown
    await app.state.session.close()
    await close_db()


app = FastAPI(lifespan=lifespan)

app.include_router(auth_router)
app.include_router(upload_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_ui() -> FileResponse:
    """Serve the OptiMotion demo UI."""
    return FileResponse(_FRONTEND_PATH)


@app.get("/logs")
async def get_logs(_user: str = Depends(get_current_user)) -> JSONResponse:
    """Return all call log entries from DB."""
    from heplers.db import get_calls
    return JSONResponse(await get_calls())


@app.get("/logs/{call_uuid}")
async def get_call_detail(call_uuid: str, _user: str = Depends(get_current_user)) -> JSONResponse:
    """Return full detail for one call: metadata + transcript + recording flag."""
    from heplers.db import get_call, get_transcript, get_recording
    call       = await get_call(call_uuid)
    transcript = await get_transcript(call_uuid)
    recording  = await get_recording(call_uuid)
    return JSONResponse({
        "call":        call,
        "transcript":  transcript,
        "recording":   recording is not None,
        "recording_file": recording.get("file_path") if recording else None,
    })


@app.get("/recording/{call_uuid}")
async def serve_recording(call_uuid: str, _user: str = Depends(get_current_user)) -> FileResponse:
    """Stream the MP3 recording file for a call."""
    from heplers.db import get_recording
    rec = await get_recording(call_uuid)
    if not rec or not rec.get("file_path"):
        raise HTTPException(status_code=404, detail="Recording not found")
    file_path = rec["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Recording file missing on disk")
    return FileResponse(file_path, media_type="audio/mpeg", filename=f"{call_uuid}.mp3")


@app.post("/start")
async def initiate_outbound_call(request: Request, _user: str = Depends(get_current_user)) -> JSONResponse:
    """Handle outbound call request and initiate call via VoBiz."""
    print("Received outbound call request")

    try:
        data = await request.json()

        # Validate request data
        if not data.get("phone_number"):
            raise HTTPException(
                status_code=400, detail="Missing 'phone_number' in the request body"
            )

        # Extract the phone number to dial
        phone_number = str(data["phone_number"])

        # Extract body data if provided
        body_data = data.get("body", {})
        print(f"Processing outbound call to {phone_number}")

        # Get server URL for answer URL
        # SERVER_BASE_URL overrides auto-detection (use when behind a proxy that strips the port)
        # e.g. SERVER_BASE_URL=https://omni.aipromptora.com:8765
        server_base_url = os.getenv("SERVER_BASE_URL")
        if server_base_url:
            host = None
            protocol = None
            base_url = server_base_url.rstrip("/")
        else:
            host = request.headers.get("host")
            if not host:
                raise HTTPException(status_code=400, detail="Unable to determine server host")
            protocol = (
                "https"
                if not host.startswith("localhost") and not host.startswith("127.0.0.1")
                else "http"
            )
            base_url = f"{protocol}://{host}"

        # Add body data as query parameters to answer URL
        answer_url = f"{base_url}/answer"
        if body_data:
            body_json = json.dumps(body_data)
            body_encoded = urllib.parse.quote(body_json)
            answer_url = f"{answer_url}?body_data={body_encoded}"

        # Initiate outbound call via VoBiz
        try:
            call_result = await make_vobiz_call(
                session=request.app.state.session,
                to_number=phone_number,
                from_number=os.getenv("VOBIZ_PHONE_NUMBER"),
                answer_url=answer_url,
            )

            # Extract call UUID from VoBiz response
            call_uuid = (
                call_result.get("request_uuid") or call_result.get("call_uuid") or "unknown"
            )

        except Exception as e:
            print(f"Error initiating VoBiz call: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initiate call: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    # Persist call to Neon DB
    from heplers.db import insert_call
    await insert_call(
        call_uuid     = call_uuid,
        timestamp     = datetime.now(),
        rider_name    = body_data.get("rider_name", ""),
        phone_number  = phone_number,
        vehicle_model = body_data.get("vehicle_model", ""),
        end_date      = body_data.get("end_date", ""),
        amount        = body_data.get("amount", ""),
        call_day      = body_data.get("call_day", ""),
        language      = body_data.get("language", ""),
        status        = "initiated",
    )

    return JSONResponse(
        {
            "call_uuid": call_uuid,
            "status": "call_initiated",
            "phone_number": phone_number,
        }
    )


@app.api_route("/answer", methods=["GET", "POST"])
async def get_answer_xml(
    request: Request,
    CallUUID: str = Query(None, description="VoBiz call UUID"),
    body_data: str = Query(None, description="JSON encoded body data"),
) -> HTMLResponse:
    """Return XML instructions for connecting call to WebSocket."""
    print("Serving answer XML for outbound call")

    # VoBiz sends CallUUID as form data on POST requests
    if request.method == "POST":
        form = await request.form()
        if not CallUUID:
            CallUUID = form.get("CallUUID")

    # Parse body data from query parameter
    parsed_body_data = {}
    if body_data:
        try:
            parsed_body_data = json.loads(body_data)
        except json.JSONDecodeError:
            print(f"Failed to parse body data: {body_data}")

    # Log call details
    if CallUUID:
        print(f"VoBiz outbound call UUID: {CallUUID}")
        if parsed_body_data:
            print(f"Body data: {parsed_body_data}")

    try:
        # Get the server host to construct WebSocket URL
        host = request.headers.get("host")
        if not host:
            raise HTTPException(status_code=400, detail="Unable to determine server host")

        protocol = (
            "https"
            if not host.startswith("localhost") and not host.startswith("127.0.0.1")
            else "http"
        )

        # Use SERVER_BASE_URL to build WS URL if set (avoids proxy stripping the port)
        server_base_url = os.getenv("SERVER_BASE_URL")
        if server_base_url:
            ws_host = server_base_url.rstrip("/").replace("https://", "").replace("http://", "")
            base_ws_url = f"wss://{ws_host}/ws"
        else:
            base_ws_url = get_websocket_url(host)

        # Add query parameters to WebSocket URL
        query_params = []

        # Add serviceHost for production
        env = os.getenv("ENV", "local").lower()
        if env == "production":
            agent_name = os.getenv("AGENT_NAME")
            org_name = os.getenv("ORGANIZATION_NAME")
            service_host = f"{agent_name}.{org_name}"
            query_params.append(f"serviceHost={service_host}")

        # Add body data if available
        if parsed_body_data:
            body_json = json.dumps(parsed_body_data)
            body_encoded = base64.b64encode(body_json.encode("utf-8")).decode("utf-8")
            query_params.append(f"body={body_encoded}")

        # Pass CallUUID so the WebSocket handler can link transcript/recordings
        if CallUUID:
            query_params.append(f"call_uuid={CallUUID}")

        # Construct final WebSocket URL
        if query_params:
            ws_url = f"{base_ws_url}?{'&amp;'.join(query_params)}"
        else:
            ws_url = base_ws_url

        # Build callback URLs for recording
        server_base_url = os.getenv("SERVER_BASE_URL")
        base_url = server_base_url.rstrip("/") if server_base_url else f"{protocol}://{host}"
        record_callback_url = f"{base_url}/recording-ready"

        # Generate XML response for VoBiz
        # <Record> must come before <Stream> to enable call recording.
        # callbackUrl is called by VoBiz when the MP3 file is ready to download.
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Record fileFormat="mp3" maxLength="3600" recordSession="true" callbackUrl="{record_callback_url}" callbackMethod="POST">
    </Record>
    <Stream bidirectional="true" audioTrack="inbound" contentType="audio/x-mulaw;rate=8000" keepCallAlive="true">
        {ws_url}
    </Stream>
</Response>"""

        return HTMLResponse(content=xml_content, media_type="application/xml")

    except Exception as e:
        print(f"Error generating answer XML: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate XML: {str(e)}")


@app.api_route("/inbound", methods=["GET", "POST"])
async def inbound_call(
    request: Request,
    CallUUID: str = Query(None),
    From: str = Query(None),
    To: str = Query(None),
) -> HTMLResponse:
    """Handle inbound call from Plivo — return Stream XML to connect to WebSocket."""
    if request.method == "POST":
        form = await request.form()
        if not CallUUID:
            CallUUID = form.get("CallUUID")
        if not From:
            From = form.get("From")
        if not To:
            To = form.get("To")

    print(f"Inbound call: {From} → {To}, UUID: {CallUUID}")

    host = request.headers.get("host")
    ws_url = f"wss://{host}/ws"

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" audioTrack="inbound" contentType="audio/x-mulaw;rate=8000" keepCallAlive="true">
        {ws_url}
    </Stream>
</Response>"""

    return HTMLResponse(content=xml_content, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    body: str = Query(None),
    serviceHost: str = Query(None),
    call_uuid: str = Query(None),
):
    """Handle WebSocket connection from VoBiz Media Streams."""
    await websocket.accept()
    print(f"WebSocket accepted — call_uuid: {call_uuid}")

    # Decode body parameter if provided
    body_data = {}
    if body:
        try:
            decoded_json = base64.b64decode(body).decode("utf-8")
            body_data = json.loads(decoded_json)
            print(f"Decoded body data: {body_data}")
        except Exception as e:
            print(f"Error decoding body parameter: {e}")

    transcript: list = []

    try:
        from heplers.bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        runner_args = WebSocketRunnerArguments(websocket=websocket, body=body_data)
        await bot(runner_args, transcript_out=transcript)

    except Exception as e:
        print(f"Error in WebSocket endpoint: {e}")
        await websocket.close()
    finally:
        # Store transcript to DB once bot finishes
        if call_uuid and transcript:
            from heplers.db import insert_transcript, update_call_analysis
            try:
                await insert_transcript(call_uuid, transcript)
                print(f"Transcript stored: {len(transcript)} turns for {call_uuid}")
            except Exception as e:
                print(f"Failed to store transcript: {e}")

            # Post-call analysis — sentiment + takeaway via gpt-4o-mini
            try:
                analysis = await _analyse_transcript(transcript)
                await update_call_analysis(call_uuid, analysis["sentiment"], analysis["takeaway"], analysis["callback"])
                print(f"Analysis stored: sentiment={analysis['sentiment']} takeaway={analysis['takeaway']!r} callback={analysis['callback']}")
            except Exception as e:
                print(f"Failed to analyse transcript: {e}")


@app.api_route("/recording-finished", methods=["GET", "POST"])
async def recording_finished(request: Request) -> HTMLResponse:
    """Called by VoBiz when recording stops — log metadata only."""
    data         = await request.form()
    call_uuid    = str(data.get("CallUUID",          "") or "")
    recording_id = str(data.get("RecordingID",       "") or "")
    duration     = str(data.get("RecordingDuration", "") or "")
    print(f"[RECORDING] finished — call: {call_uuid}, id: {recording_id}, duration: {duration}s")
    return HTMLResponse(content="<Response></Response>", media_type="application/xml")


@app.api_route("/recording-ready", methods=["GET", "POST"])
async def recording_ready(request: Request) -> HTMLResponse:
    """Called by VoBiz when the MP3 file is ready — download and store to DB."""
    data          = await request.form()
    recording_url = str(data.get("RecordUrl",  "") or "")
    recording_id  = str(data.get("RecordingID", "") or "")
    call_uuid     = str(data.get("CallUUID",    "") or "")

    print(f"[RECORDING] ready — call: {call_uuid}, id: {recording_id}, url: {recording_url}")

    if recording_url and recording_id:
        try:
            os.makedirs("recordings", exist_ok=True)
            file_path = f"recordings/{recording_id}.mp3"

            auth_id    = os.getenv("VOBIZ_AUTH_ID",    "")
            auth_token = os.getenv("VOBIZ_AUTH_TOKEN", "")
            headers: dict[str, str] = {"X-Auth-ID": auth_id, "X-Auth-Token": auth_token}

            async with aiohttp.ClientSession() as session:
                async with session.get(recording_url, headers=headers) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        with open(file_path, "wb") as f:
                            f.write(audio_data)
                        print(f"[RECORDING] downloaded → {file_path} ({len(audio_data)} bytes)")

                        # Store to DB
                        if call_uuid:
                            from heplers.db import insert_recording
                            await insert_recording(
                                call_uuid     = call_uuid,
                                recording_id  = recording_id,
                                recording_url = recording_url,
                                file_path     = file_path,
                            )
                    else:
                        err = await resp.text()
                        print(f"[RECORDING] download failed: HTTP {resp.status} — {err}")
        except Exception as e:
            print(f"[RECORDING] error: {e}")

    return HTMLResponse(content="<Response></Response>", media_type="application/xml")


# ----------------- Main ----------------- #


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7866)