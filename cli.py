"""Terminal chat interface. Run: python cli.py

Commands:  /reset  start a new conversation   /context  show what the bot remembers
           /quit   exit
"""
import sys

from app import build_service
from llm import LLMError


def main() -> None:
    try:
        service = build_service()
    except ValueError as exc:
        raise SystemExit(f"Config error: {exc}")

    print("Byte - your coding mentor. Type /quit to exit, /reset to start over, /context to inspect memory.\n")
    conv_id = None
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            if conv_id:
                service.reset(conv_id)
            conv_id = None
            print("(new conversation)\n")
            continue
        if text == "/context":
            conv = service.get(conv_id) if conv_id else None
            if not conv:
                print("(no conversation yet)\n")
            else:
                print(f"messages in window: {len(conv.messages)}")
                print(f"summary: {conv.summary or '(none yet)'}\n")
            continue

        try:
            conv_id, reply = service.send(conv_id, text)
            print(f"\nbyte > {reply}\n")
        except LLMError as err:
            print(f"\n[error] {err.user_message}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
