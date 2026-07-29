# Telegram Data Analyst Bot

FastAPI + Groq (`llama-3.3-70b-versatile`) agent that answers data-analysis questions
sent via Telegram and replies with `{"answer": ..., "log_url": ...}`.

## Local setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # then fill in real values
```

Run locally with a tunnel (e.g. ngrok) if you want to test the webhook before deploying:

```bash
uvicorn main:app --reload --port 8000
ngrok http 8000
```

## Deploy to Render

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variables: `TELEGRAM_BOT_TOKEN`, `GROQ_API_KEY`, `PUBLIC_BASE_URL` (your Render URL)

## Register the Telegram webhook

After deploying, run once (replace values):

```bash
curl "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://<your-render-url>/webhook"
```

## Log

Every incoming message, tool call, and final answer is appended to `run.jsonl` and served
publicly at `GET /run.jsonl` — this is the URL you put in `PUBLIC_BASE_URL`.

**Note:** Render's free-tier filesystem is ephemeral and resets on redeploy/restart, but
persists for the lifetime of a running instance, which is sufficient for a live grading session.
