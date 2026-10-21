"""Flask backend: JSON API + static frontend."""
import logging
import os

from flask import Flask, jsonify, request, send_from_directory

import config
from conversation import ChatService
from llm import GeminiClient, LLMError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def build_service() -> ChatService:
    llm = GeminiClient(
        api_key=config.GEMINI_API_KEY,
        model=config.GEMINI_MODEL,
        timeout_s=config.REQUEST_TIMEOUT_SECONDS,
        max_retries=config.MAX_RETRIES,
    )
    return ChatService(
        llm,
        max_recent=config.MAX_RECENT_MESSAGES,
        summarize_batch=config.SUMMARIZE_BATCH,
        max_conversations=config.MAX_CONVERSATIONS,
    )


def create_app(service: ChatService | None = None) -> Flask:
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
    app.config["SERVICE"] = service or build_service()

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        text = (data.get("message") or "").strip()
        conv_id = data.get("conversation_id")

        if not text:
            return jsonify(error="Please type a message first."), 400
        if len(text) > config.MAX_MESSAGE_CHARS:
            return jsonify(
                error=f"Message too long (max {config.MAX_MESSAGE_CHARS} characters)."
            ), 400

        try:
            conv_id, reply = app.config["SERVICE"].send(conv_id, text)
        except LLMError as err:
            return jsonify(error=err.user_message), err.status
        except Exception:  # noqa: BLE001
            log.exception("Unhandled error in /api/chat")
            return jsonify(error="Unexpected server error. Please try again."), 500

        conv = app.config["SERVICE"].get(conv_id)
        return jsonify(
            conversation_id=conv_id,
            reply=reply,
            context={
                "messages_in_window": len(conv.messages),
                "has_summary": bool(conv.summary),
            },
        )

    @app.post("/api/reset")
    def reset():
        data = request.get_json(silent=True) or {}
        if data.get("conversation_id"):
            app.config["SERVICE"].reset(data["conversation_id"])
        return jsonify(ok=True)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    return app


if __name__ == "__main__":
    try:
        application = create_app()
    except ValueError as exc:  # missing API key
        raise SystemExit(f"Config error: {exc}")
    application.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=False)
