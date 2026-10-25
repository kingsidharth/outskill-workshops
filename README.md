# outskill-workshops

Hands-on workshop material for building AI agents and demo applications.

## Demo App

The functional demo for the OpenAI Agents SDK workshop lives in `demo-app/`.

Run it locally:

```bash
cd demo-app
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` to try the IndiGo customer service chatbot and the OpenAI voice interview demo.

## CrewAI bootcamp notebooks

Run the first setup cell in each notebook before the remaining cells. These cells use
`%pip` so dependencies install into the selected notebook kernel. Restart the kernel
after installing if CrewAI was already imported.

CrewAI's tools extra does not install the optional Exa SDK. The setup cells and demo
requirements explicitly include `exa-py`; without it, constructing `EXASearchTool`
can prompt to install a package and prevent the demo server from starting.

The root `.venv` and `demo-app/.venv` are separate environments. Select the intended
kernel for notebooks and use the explicit demo interpreter in the commands above.
