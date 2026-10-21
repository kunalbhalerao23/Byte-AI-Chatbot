# Byte - an LLM-powered coding mentor chatbot

A multi-turn chatbot backed by the **Google Gemini API**, with real context-window
management, a defined persona, graceful error handling, a web UI and a CLI.

## Features

| Requirement | Where |
|---|---|
| Backend endpoint that calls an LLM | `POST /api/chat` in `app.py` -> `llm.py` |
| Multi-turn history per conversation | `ChatService` / `Conversation` in `conversation.py` |
| Context-window management | `ContextManager` (sliding window + running summary) |
| System prompt (role/personality) | `prompts.py` - "Byte", a beginner coding mentor |
| Error handling (failure / timeout / rate limit) | `llm.py` (timeouts, retries, friendly messages) + UI error bubbles |
| API key in env vars, never committed | `.env` (git-ignored), `.env.example` template |
| Frontend / CLI | `static/index.html`, `cli.py` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# edit .env and paste your key from https://aistudio.google.com/apikey

python app.py               # web UI at http://127.0.0.1:5000
# or
python cli.py               # terminal chat
```

Run the offline tests (no API key or network needed):

```bash
python -m pytest -q
```

## How context management works

LLM APIs are stateless, so every request re-sends the conversation. To keep that
bounded, two layers are used (see `conversation.py`):

1. **Sliding window** - the last `MAX_RECENT_MESSAGES` (default 10) messages are
   always sent word-for-word.
2. **Running summary** - once history exceeds the window by `SUMMARIZE_BATCH`
   (default 4) messages, the oldest messages are condensed by the LLM into a short
   summary. The summary is prepended to the system prompt, so facts from early in
   the chat (your name, your project) are still remembered.

Summarising only happens every few turns (not every turn) to save API calls. If the
summary call fails, the bot falls back to plain truncation, so the chat never breaks.
The retained window always starts on a user message, as the API expects.

## Error handling

- Timeouts (`REQUEST_TIMEOUT_SECONDS`), network failures, rate limits (429) and 5xx
  errors are retried with exponential backoff (`MAX_RETRIES`).
- If it still fails, the user sees a clear message in the chat - never a broken page.
- A failed turn is **rolled back** from history, so the user can just resend.
- Invalid API key problems are reported generically; internal details go to server logs only.
- Empty / oversized messages are rejected with a 400.

## API

`POST /api/chat` `{ "message": "...", "conversation_id": "optional" }`
-> `{ "conversation_id", "reply", "context": { "messages_in_window", "has_summary" } }`

`POST /api/reset` `{ "conversation_id": "..." }` clears a conversation.

## Manual 5+ turn check (do this before submitting)

1. "Hi, my name is <you> and I'm learning Python."
2. "What's a list comprehension?"
3. "Show me one that squares numbers."
4. "Thanks. What was my name again?"  -> should answer correctly
5. "And what language am I learning?"  -> should answer correctly

To see summarisation kick in, set `MAX_RECENT_MESSAGES=4` and `SUMMARIZE_BATCH=2` in
`.env`, chat for ~8 turns, then ask about something from turn 1. In the web UI the
footer shows `Memory: last N messages + summary`; in the CLI type `/context`.

## Notes

- Conversations are stored in memory (lost on restart) - fine for this scope;
  swap `ChatService`'s dict for Redis/SQLite to persist.
- If `gemini-2.5-flash` is retired or unavailable to your key, set `GEMINI_MODEL`
  in `.env` to any current Gemini model listed in Google AI Studio.
