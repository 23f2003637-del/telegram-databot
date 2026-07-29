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


def tool_web_search(query: str) -> str:
    """Search the web via DuckDuckGo's HTML endpoint and return titles + URLs."""
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for a in soup.select("a.result__a")[:8]:
            title = a.get_text(strip=True)
            href = a.get("href", "")
            results.append(f"{title} -- {href}")
        if not results:
            return "No results found."
        return "\n".join(results)
    except Exception as e:
        return f"ERROR searching '{query}': {e}"


def tool_run_python(code: str) -> str:
    """
    Execute Python code in a sandboxed namespace with pandas/requests available.
    The code should print() whatever result it wants returned.
    """
    stdout = io.StringIO()

    class _SafeRequests:
        """Wraps requests.get/post to force a default timeout if the agent forgets one."""
        @staticmethod
        def get(*args, **kwargs):
            kwargs.setdefault("timeout", 20)
            return requests.get(*args, **kwargs)

        @staticmethod
        def post(*args, **kwargs):
            kwargs.setdefault("timeout", 20)
            return requests.post(*args, **kwargs)

    local_ns: dict[str, Any] = {"pd": pd, "requests": _SafeRequests, "json": json}
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
            "name": "web_search",
            "description": "Search the web for a query and get back a list of page titles and URLs. Use this FIRST when you don't already know the exact URL for a dataset -- never guess a domain name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
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

TOOL_IMPL = {"web_search": tool_web_search, "fetch_url": tool_fetch_url, "run_python": tool_run_python}

SYSTEM_PROMPT = """You are a careful data analyst agent. You will be given a data-analysis \
question via a Telegram message. The message will tell you EXACTLY what JSON shape to reply \
with for the "answer" field.

CRITICAL RULES:
1. NEVER fabricate, guess, hardcode, or assume a numeric/factual answer -- not even as a \
last resort after tool errors. If fetch_url or run_python fail, that is a signal to try a \
DIFFERENT approach, never a signal to fall back on what you already "know". Answering from \
memory instead of verified tool output is a failure condition, even if the number happens to \
be correct.
2. NEVER guess a URL or domain name from memory (e.g. assuming a government site is on \
.nic.in vs .gov.in). Use web_search first to find the correct, real URL before calling \
fetch_url. Guessed domains frequently fail to resolve and waste attempts.
2. When writing code for run_python, ALWAYS use real newlines between statements (write \
multi-line code, not everything crammed onto one line with semicolons). Never place a '#' \
comment on the same line before other code you intend to execute -- anything after '#' is \
ignored until the next newline, which will silently delete your remaining code.
3. After every run_python call, check the returned output carefully. If it says \
"(no output; use print() to return results)" or shows an error, your code did NOT produce a \
usable result -- fix the code and retry with a different strategy rather than proceeding as if \
it worked.
4. If fetch_url returns a webpage (HTML) rather than raw data, parse it (e.g. via \
BeautifulSoup or pandas.read_html, both available) to find links to actual data files \
(CSV/XLSX/JSON/API endpoints) and fetch those instead of trying to parse navigation HTML as if \
it were a data table.
5. IMPORTANT: mospi.gov.in is a JavaScript-rendered single-page app -- fetching it with \
fetch_url only returns an empty HTML shell with no real data, no matter how you parse it. For \
MOSPI-sourced statistics (health, economic, social indicators), prefer searching for the same \
data published as a plain-HTML Press Information Bureau release at pib.gov.in (search-engine-\
indexable, e.g. via a Google-style query embedded in a fetch_url to a search results page, or \
by trying likely pib.gov.in URLs), or other plain-HTML government/news sources that actually \
contain the figures in the page text, rather than repeatedly retrying mospi.gov.in itself.
6. If, after several genuine attempts with different strategies, you truly cannot retrieve \
verifiable data, say so plainly in the answer field (e.g. null or a short explanatory string) \
rather than inventing a plausible-looking number.
7. Once you have a real, verified answer, respond with ONLY a single JSON object and nothing \
else (no markdown fences, no explanation), with exactly two top-level keys:
   - "answer": shaped exactly as the incoming question asked for.
   - "log_url": the literal string PLACEHOLDER_LOG_URL (it will be substituted automatically).
8. Never include any text outside that single JSON object in your final reply.
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


def _coerce_nulls(obj):
    """Recursively convert string 'null'/'none' (any case) into real JSON null."""
    if isinstance(obj, dict):
        return {k: _coerce_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_nulls(v) for v in obj]
    if isinstance(obj, str) and obj.strip().lower() in ("null", "none"):
        return None
    return obj


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
        obj = _coerce_nulls(obj)
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
