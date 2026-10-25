# Outskill Workshops Demo App

Functional UI demos for the OpenAI Agents SDK workshop notebooks.

## Demos

- `IndiGo Assistant` shows the customer service chatbot from `openai_agents_sdk/notebooks/01_customer_service_chatbot.ipynb`.
- `OpenAI Assessment` shows the audio-to-audio interview flow from `openai_agents_sdk/notebooks/02_job_interview_agent.ipynb`.

## Run Locally

```bash
cd demo-app  # from the repository root
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`.

Use Python 3.11 or 3.12. Always use the same `.venv/bin/python` for installation
and startup. The notebooks can use a different kernel, so a package installed in
the root `.venv` is not available in this demo's `.venv` automatically.

Configure `OPENAI_API_KEY` (agents and interview speech), `EXA_API_KEY`, and
`GEMINI_API_KEY` (podcast speech; `GOOGLE_API_KEY` is also accepted)
in the runtime environment before starting. The app also supports its existing
repository-root dotenv configuration. Exa needs both the `exa-py` SDK (included in
requirements) and an API key. `python-multipart` is included for voice uploads.

Document search is optional and disabled by default. IndiGo uses its keyword FAQ
fallback and the interview uses web search. To enable document search, explicitly
set `ENABLE_DOCUMENT_SEARCH=true` before startup; this uploads the bundled FAQ and
resume PDFs to OpenAI vector stores. Finance and podcast runs call Exa for research.

## Pages

- `/` — AI agent showcase home
- `/indigo.html` — IndiGo assistant landing page
- `/indigo-chat.html` — IndiGo chatbot
- `/assessment.html` — OpenAI assessment dashboard
- `/interview.html` — live voice interview UI
- `/finance.html` — finance research pipeline
- `/podcaster.html` — podcast research, script, and voice pipeline

## APIs

- `POST /api/indigo/chat`
- `POST /api/interview/start`
- `POST /api/interview/message`
- `POST /api/interview/voice`
- `POST /api/tts`

## Verify setup

From `demo-app/`:

```bash
OTEL_SDK_DISABLED=true .venv/bin/python -m unittest discover -s tests -v
```

The checks construct both Exa researcher agents, serve the demo pages, and verify
that document uploads happen only after explicit opt-in (with upload calls mocked).

If this environment was created with uv, install using
`uv pip install --python .venv/bin/python -r requirements.txt` instead of pip.

Podcast voice uses Gemini multi-speaker TTS: Joe uses Puck and Jane uses Kore.
Long dialogue is split into bounded requests and the WAV audio is joined in order. The full transcript is retained, and PCM output is wrapped as WAV for
browser playback. Default model: `gemini-3.1-flash-tts-preview`; override with
`GEMINI_TTS_MODEL` if needed. Podcast speech does not require an OpenRouter key.
