import asyncio
import base64
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv("../.env")

from app_agents.indigo_agent import BookingContext, init_vector_store, run_chat
from app_agents.interview_agent import init_resume_store, send_message, start_session
from app_agents.finance_agent import run_analysis
from app_agents.podcaster_agent import run_podcast

BASE_DIR = Path(__file__).parent
FAQ_PDF = BASE_DIR.parent / "openai_agents_sdk" / "docs" / "IndiGo_FAQ.pdf"
RESUME_PDF = BASE_DIR.parent / "openai_agents_sdk" / "docs" / "IshanDuttaResume.pdf"

_indigo_sessions: dict[str, dict] = {}
_finance_jobs: dict[str, dict] = {}
_podcaster_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if FAQ_PDF.exists():
        print(f"Initializing IndiGo FAQ vector store from {FAQ_PDF}...")
        await init_vector_store(str(FAQ_PDF))
    else:
        print(
            f"Warning: IndiGo FAQ PDF not found at {FAQ_PDF}. Using keyword fallback."
        )

    if RESUME_PDF.exists():
        print(f"Initializing resume vector store from {RESUME_PDF}...")
        await init_resume_store(str(RESUME_PDF))
    else:
        print("Resume PDF not found. Interview agent will use web search only.")

    yield


app = FastAPI(title="Outskill Workshops Demo", lifespan=lifespan)

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/{page}.html")
async def serve_page(page: str):
    file_path = STATIC_DIR / f"{page}.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(str(file_path))


# ---------- IndiGo Chat API ----------


class IndigoChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class IndigoChatResponse(BaseModel):
    reply: str
    agent: str
    handoffs: list[str]
    blocked: bool
    session_id: str


@app.post("/api/indigo/chat", response_model=IndigoChatResponse)
async def indigo_chat(req: IndigoChatRequest):
    session_id = req.session_id or "default"

    if session_id not in _indigo_sessions:
        _indigo_sessions[session_id] = {
            "history": [],
            "context": BookingContext(),
        }

    session = _indigo_sessions[session_id]

    result = await run_chat(
        user_message=req.message,
        history=session["history"],
        context=session["context"],
    )

    session["history"] = result["history"]
    session["context"] = result["context"]

    return IndigoChatResponse(
        reply=result["reply"],
        agent=result["agent"],
        handoffs=result["handoffs"],
        blocked=result["blocked"],
        session_id=session_id,
    )


# ---------- Interview API ----------


class InterviewStartResponse(BaseModel):
    session_id: str
    reply: str
    agent: str
    ended: bool


class InterviewMessageRequest(BaseModel):
    session_id: str
    message: str


class InterviewMessageResponse(BaseModel):
    reply: str
    agent: str
    handoffs: list[str]
    ended: bool


@app.post("/api/interview/start", response_model=InterviewStartResponse)
async def interview_start():
    result = await start_session()
    return InterviewStartResponse(
        session_id=result["session_id"],
        reply=result["reply"],
        agent=result["agent"],
        ended=result["ended"],
    )


@app.post("/api/interview/message", response_model=InterviewMessageResponse)
async def interview_message(req: InterviewMessageRequest):
    result = await send_message(req.session_id, req.message)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return InterviewMessageResponse(
        reply=result["reply"],
        agent=result["agent"],
        handoffs=result.get("handoffs", []),
        ended=result["ended"],
    )


# ---------- Voice Interview API ----------

_openai_client = OpenAI()


@app.post("/api/interview/voice")
async def interview_voice(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    audio_bytes = await audio.read()

    transcription = await asyncio.to_thread(
        lambda: _openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.webm", audio_bytes, audio.content_type or "audio/webm"),
        )
    )
    user_text = transcription.text.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="No speech detected")

    result = await send_message(session_id, user_text)

    tts_response = await asyncio.to_thread(
        lambda: _tts_client.audio.speech.create(
            model=_TTS_MODEL,
            voice="nova",
            input=result["reply"],
            response_format="mp3",
        )
    )
    reply_audio_b64 = base64.b64encode(tts_response.content).decode()

    return {
        "user_transcript": user_text,
        "reply_text": result["reply"],
        "reply_audio": reply_audio_b64,
        "agent": result["agent"],
        "handoffs": result.get("handoffs", []),
        "ended": result["ended"],
    }


# ---------- TTS utility ----------

_tts_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
_TTS_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"


class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"


@app.post("/api/tts")
async def tts(req: TTSRequest):
    tts_response = await asyncio.to_thread(
        lambda: _tts_client.audio.speech.create(
            model=_TTS_MODEL,
            voice=req.voice,
            input=req.text,
            response_format="mp3",
        )
    )
    audio_b64 = base64.b64encode(tts_response.content).decode()
    return {"audio": audio_b64}


# ---------- Finance Pipeline API ----------


class FinanceStartRequest(BaseModel):
    company: str


class FinanceStartResponse(BaseModel):
    job_id: str


class FinanceStatusResponse(BaseModel):
    status: str
    steps: dict[str, str]
    result: dict | None = None
    error: str | None = None


def _finance_callback(job_id: str):
    def cb(step: str, status: str):
        _finance_jobs[job_id]["steps"][step] = status
    return cb


async def _run_finance_job(job_id: str, company: str):
    try:
        result = await asyncio.to_thread(
            run_analysis, company, _finance_callback(job_id)
        )
        _finance_jobs[job_id]["status"] = "completed"
        _finance_jobs[job_id]["result"] = result
    except Exception as e:
        _finance_jobs[job_id]["status"] = "failed"
        _finance_jobs[job_id]["error"] = str(e)


@app.post("/api/finance/start", response_model=FinanceStartResponse)
async def finance_start(req: FinanceStartRequest):
    job_id = str(uuid.uuid4())
    _finance_jobs[job_id] = {
        "status": "running",
        "steps": {},
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_finance_job(job_id, req.company))
    return FinanceStartResponse(job_id=job_id)


@app.get("/api/finance/status/{job_id}", response_model=FinanceStatusResponse)
async def finance_status(job_id: str):
    if job_id not in _finance_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _finance_jobs[job_id]
    return FinanceStatusResponse(
        status=job["status"],
        steps=job["steps"],
        result=job["result"],
        error=job["error"],
    )


# ---------- Podcaster Pipeline API ----------


class PodcasterStartRequest(BaseModel):
    topic: str


class PodcasterStartResponse(BaseModel):
    job_id: str


class PodcasterStatusResponse(BaseModel):
    status: str
    steps: dict[str, str]
    result: dict | None = None
    error: str | None = None


def _podcaster_callback(job_id: str):
    def cb(step: str, status: str):
        _podcaster_jobs[job_id]["steps"][step] = status
    return cb


def _parse_script_segments(script_text: str) -> list[dict]:
    segments = []
    for line in script_text.split("\n"):
        line = line.strip()
        m = re.match(r"\*{0,2}(Joe|Jane)\*{0,2}:\s*(.*)", line, re.IGNORECASE)
        if m:
            speaker = m.group(1).capitalize()
            text = m.group(2).strip().strip("*").strip()
            if text and len(text) > 4:
                segments.append({"speaker": speaker, "text": text})
    return segments


async def _tts_segment(text: str, voice: str) -> bytes:
    resp = await asyncio.to_thread(
        lambda: _tts_client.audio.speech.create(
            model=_TTS_MODEL, voice=voice, input=text, response_format="mp3"
        )
    )
    return resp.content


async def _run_podcaster_job(job_id: str, topic: str):
    try:
        result = await asyncio.to_thread(
            run_podcast, topic, _podcaster_callback(job_id)
        )

        _podcaster_jobs[job_id]["steps"]["voice"] = "running"
        voice_map = {"Joe": "onyx", "Jane": "nova"}
        segments = _parse_script_segments(result["script"])[:50]

        tasks = [_tts_segment(seg["text"], voice_map.get(seg["speaker"], "alloy")) for seg in segments]
        audio_contents = await asyncio.gather(*tasks)

        result["segments"] = [
            {"speaker": seg["speaker"], "text": seg["text"], "audio": base64.b64encode(content).decode()}
            for seg, content in zip(segments, audio_contents)
        ]
        _podcaster_jobs[job_id]["steps"]["voice"] = "done"
        _podcaster_jobs[job_id]["status"] = "completed"
        _podcaster_jobs[job_id]["result"] = result
    except Exception as e:
        _podcaster_jobs[job_id]["status"] = "failed"
        _podcaster_jobs[job_id]["error"] = str(e)


@app.post("/api/podcaster/start", response_model=PodcasterStartResponse)
async def podcaster_start(req: PodcasterStartRequest):
    job_id = str(uuid.uuid4())
    _podcaster_jobs[job_id] = {
        "status": "running",
        "steps": {},
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_podcaster_job(job_id, req.topic))
    return PodcasterStartResponse(job_id=job_id)


@app.get("/api/podcaster/status/{job_id}", response_model=PodcasterStatusResponse)
async def podcaster_status(job_id: str):
    if job_id not in _podcaster_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _podcaster_jobs[job_id]
    return PodcasterStatusResponse(
        status=job["status"],
        steps=job["steps"],
        result=job["result"],
        error=job["error"],
    )

