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

Configure `OPENAI_API_KEY`, `OPENROUTER_API_KEY` (voice generation), and `EXA_API_KEY`
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
