"""Conversation state + context-window management.

Strategy (two layers):
  1. Sliding window  - only the last MAX_RECENT_MESSAGES messages are sent verbatim.
  2. Running summary - when history grows past the window, the oldest messages are
     condensed by the LLM into a short summary that is prepended to the system prompt.
     If summarising fails, we fall back to plain truncation so chat never breaks.
"""
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from llm import LLMError
from prompts import SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT

log = logging.getLogger(__name__)


@dataclass
class Conversation:
    id: str
    messages: list = field(default_factory=list)  # [{"role", "content"}]
    summary: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ContextManager:
    def __init__(self, llm, max_recent: int = 10, summarize_batch: int = 4):
        self.llm = llm
        self.max_recent = max(2, max_recent)
        self.summarize_batch = max(1, summarize_batch)

    def compress_if_needed(self, conv: Conversation) -> None:
        """Fold the oldest messages into conv.summary once history exceeds the window."""
        if len(conv.messages) <= self.max_recent + self.summarize_batch:
            return

        cut = len(conv.messages) - self.max_recent
        # Keep the retained window starting on a user turn (models expect that).
        while cut < len(conv.messages) and conv.messages[cut]["role"] != "user":
            cut += 1
        older, recent = conv.messages[:cut], conv.messages[cut:]
        if not older:
            return

        try:
            conv.summary = self._summarize(conv.summary, older)
            log.info("Summarised %d old messages for %s", len(older), conv.id)
        except LLMError:
            # Degrade gracefully: plain truncation, keep any existing summary.
            log.warning("Summarisation failed; falling back to truncation for %s", conv.id)
        conv.messages = recent

    def build_request(self, conv: Conversation) -> tuple[str, list]:
        """Return (system_instruction, messages) to send to the model."""
        system = SYSTEM_PROMPT
        if conv.summary:
            system += (
                "\n\nSUMMARY OF EARLIER CONVERSATION "
                "(older messages were condensed to save space; treat as true context):\n"
                + conv.summary
            )
        return system, list(conv.messages)

    def _summarize(self, existing: str, older: list) -> str:
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in older
        )
        prompt = (
            f"Existing summary:\n{existing or '(none)'}\n\n"
            f"Older messages to fold in:\n{transcript}"
        )
        return self.llm.generate(
            SUMMARY_SYSTEM_PROMPT, [{"role": "user", "content": prompt}]
        )


class ChatService:
    """Owns all conversations (in memory) and orchestrates one chat turn."""

    def __init__(self, llm, max_recent: int = 10, summarize_batch: int = 4,
                 max_conversations: int = 1000):
        self.llm = llm
        self.context = ContextManager(llm, max_recent, summarize_batch)
        self._convs: "OrderedDict[str, Conversation]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_conversations = max_conversations

    # --------------------------------------------------------------- store
    def _get_or_create(self, conv_id: str | None) -> Conversation:
        with self._lock:
            if conv_id and conv_id in self._convs:
                self._convs.move_to_end(conv_id)
                return self._convs[conv_id]
            conv = Conversation(id=uuid.uuid4().hex)
            self._convs[conv.id] = conv
            while len(self._convs) > self._max_conversations:
                self._convs.popitem(last=False)
            return conv

    def get(self, conv_id: str) -> Conversation | None:
        with self._lock:
            return self._convs.get(conv_id)

    def reset(self, conv_id: str) -> None:
        with self._lock:
            self._convs.pop(conv_id, None)

    # ---------------------------------------------------------------- chat
    def send(self, conv_id: str | None, text: str) -> tuple[str, str]:
        """Process one user message. Returns (conversation_id, reply).

        Raises LLMError (with a user-safe message) if the model call fails; in that
        case the user's message is NOT kept in history, so they can simply retry.
        """
        conv = self._get_or_create(conv_id)
        with conv.lock:
            conv.messages.append({"role": "user", "content": text})
            try:
                self.context.compress_if_needed(conv)
                system, messages = self.context.build_request(conv)
                reply = self.llm.generate(system, messages)
            except Exception:
                # Roll back the dangling user message so history stays consistent.
                if conv.messages and conv.messages[-1] == {"role": "user", "content": text}:
                    conv.messages.pop()
                raise
            conv.messages.append({"role": "assistant", "content": reply})
        return conv.id, reply
