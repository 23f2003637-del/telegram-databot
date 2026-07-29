import os
import json
import time
import uuid
import traceback
import contextlib
import io
from typing import Any

import requests
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "llama-3.3-70b-versatile"
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

LOG_PATH = os.path.join(os.path.dirname(__file__), "run.jsonl")
# Set this once you know your deployed Render URL, used to build log_url in replies.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://your-app.onrender.com")

client = Groq(api_key=GROQ_API_KEY)
app = FastAPI()

# In-memory per-chat conversation history: {chat_id: [ {role, content}, ... ]}
CHAT_HISTORY: dict[int, list[dict]] = {}
MAX_HISTORY_TURNS = 6  # keep last N user/assistant turns per chat

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_event(event: dict):
    event = {"ts": time.time(), **event}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# ---------------------------------------------------------------------------
# Tools the LLM can call
# ---------------------------------------------------------------------------
def tool_fetch_url(url: str) -> str:
    """Fetch a URL and return its text content (truncated)."""
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = resp.text
        # Truncate to keep tokens reasonable; full data can be re-fetched by run_python if needed.
        return text[:20000]
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


def tool_run_python(code: str) -> str:
    """
    Execute Python code in a sandboxed namespace with pandas/requests available.
    The code should print() whatever result it wants returned.
    """
    stdout = io.StringIO()
    local_ns: dict[str, Any] = {"pd": pd, "requests": requests, "json": json}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": __builtins__}, local_ns)
        output = stdout.getvalue()
        return output[-8000:] if output else "(no output; use print() to return results)"
    except Exception:
        return f"ERROR:\n{traceback.format_exc()[-4000:]}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the raw text/HTML/CSV/JSON content of a public URL (e.g. a MOSPI dataset page or a direct file link).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code (pandas/requests available) to compute an answer. Use print() to output results you want to see.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        },
    },
]

TOOL_IMPL = {"fetch_url": tool_fetch_url, "run_python": tool_run_python}

SYSTEM_PROMPT = """You are a careful data analyst agent. You will be given a data-analysis \
question via a Telegram message. The message will tell you EXACTLY what JSON shape to reply \
with for the "answer" field.

Rules:
1. Use the fetch_url and run_python tools as needed to actually retrieve and compute the answer. \
Do not guess or fabricate numbers you could compute or look up.
2. Once you have the final answer, respond with ONLY a single JSON object and nothing else \
(no markdown fences, no explanation), with exactly two top-level keys:
   - "answer": shaped exactly as the incoming question asked for.
   - "log_url": the literal string PLACEHOLDER_LOG_URL (it will be substituted automatically).
3. Never include any text outside that single JSON object in your final reply.
"""


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------
def run_agent(chat_id: int, user_text: str) -> str:
    history = CHAT_HISTORY.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    # trim history
    trimmed = history[-(MAX_HISTORY_TURNS * 2):]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed

    log_event({"chat_id": chat_id, "type": "incoming_message", "text": user_text})

    for step in range(8):  # cap tool-call loop
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = TOOL_IMPL.get(fn_name, lambda **_: "ERROR: unknown tool")(**args)
                log_event({"chat_id": chat_id, "type": "tool_call", "tool": fn_name,
                           "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
            continue  # let the model see tool results and continue

        # No tool calls -> this is the final answer
        final_text = (msg.content or "").strip()
        history.append({"role": "assistant", "content": final_text})
        log_event({"chat_id": chat_id, "type": "final_answer_raw", "text": final_text})
        return final_text

    # Loop cap hit; return whatever we have
    return json.dumps({"answer": None, "log_url": "PLACEHOLDER_LOG_URL"})


def finalize_reply(raw_text: str) -> str:
    """Ensure the reply is valid JSON with a real log_url substituted in."""
    log_url = f"{PUBLIC_BASE_URL}/run.jsonl"
    try:
        # Strip accidental markdown fences if the model added them
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        obj = json.loads(cleaned)
        obj["log_url"] = log_url
        return json.dumps(obj)
    except Exception:
        # Fallback: wrap raw text as the answer so we still return valid JSON
        return json.dumps({"answer": raw_text, "log_url": log_url})


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------
def send_message(chat_id: int, text: str):
    requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=20)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/run.jsonl")
def get_log():
    if not os.path.exists(LOG_PATH):
        return PlainTextResponse("", media_type="application/json")
    with open(LOG_PATH) as f:
        content = f.read()
    return PlainTextResponse(content, media_type="application/json")


@app.post("/webhook")
async def webhook(request: Request):
    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message or "text" not in message:
        return {"ok": True}

    chat_id = message["chat"]["id"]
    text = message["text"]

    try:
        raw = run_agent(chat_id, text)
        final = finalize_reply(raw)
    except Exception:
        err = traceback.format_exc()
        log_event({"chat_id": chat_id, "type": "error", "trace": err})
        final = json.dumps({"answer": None, "log_url": f"{PUBLIC_BASE_URL}/run.jsonl"})

    send_message(chat_id, final)
    log_event({"chat_id": chat_id, "type": "outgoing_reply", "text": final})
    return {"ok": True}
