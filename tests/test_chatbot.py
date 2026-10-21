"""Offline tests: a FakeLLM stands in for Gemini, so no API key or network is needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from conversation import ChatService  # noqa: E402
from llm import LLMError  # noqa: E402
from prompts import SUMMARY_SYSTEM_PROMPT, SYSTEM_PROMPT  # noqa: E402


class FakeLLM:
    """Records every call. Summary calls return a canned summary."""

    def __init__(self):
        self.calls = []
        self.fail_next = None

    def generate(self, system_instruction, messages):
        self.calls.append((system_instruction, [dict(m) for m in messages]))
        if self.fail_next:
            err, self.fail_next = self.fail_next, None
            raise err
        if system_instruction == SUMMARY_SYSTEM_PROMPT:
            return "SUMMARY: user is Kunal, learning Python."
        return f"reply-{len(messages)}"


def make_service(**kw):
    llm = FakeLLM()
    return llm, ChatService(llm, **kw)


def test_history_is_sent_each_turn_so_model_can_see_earlier_messages():
    llm, svc = make_service(max_recent=10, summarize_batch=4)
    cid, _ = svc.send(None, "My name is Kunal")
    svc.send(cid, "I like Python")
    svc.send(cid, "What is my name?")
    _, last_messages = llm.calls[-1]
    contents = [m["content"] for m in last_messages]
    assert "My name is Kunal" in contents          # 2-3 messages earlier is still present
    assert last_messages[-1]["content"] == "What is my name?"
    assert [m["role"] for m in last_messages] == ["user", "assistant", "user", "assistant", "user"]


def test_system_prompt_is_applied():
    llm, svc = make_service()
    svc.send(None, "hi")
    system, _ = llm.calls[0]
    assert system.startswith(SYSTEM_PROMPT)
    assert "Byte" in SYSTEM_PROMPT


def test_old_messages_are_summarised_and_window_is_bounded():
    llm, svc = make_service(max_recent=6, summarize_batch=2)
    cid = None
    for i in range(12):
        cid, _ = svc.send(cid, f"message {i}")
    conv = svc.get(cid)
    assert conv.summary.startswith("SUMMARY")
    assert len(conv.messages) <= 6 + 2 + 1          # bounded window
    assert conv.messages[0]["role"] == "user"        # window starts on a user turn
    # The latest real chat call carries the summary in the system prompt.
    system, msgs = [c for c in llm.calls if c[0] != SUMMARY_SYSTEM_PROMPT][-1]
    assert "SUMMARY OF EARLIER CONVERSATION" in system
    assert len(msgs) <= 6 + 2 + 1


def test_no_summarisation_for_short_conversations():
    llm, svc = make_service(max_recent=10, summarize_batch=4)
    cid = None
    for i in range(5):
        cid, _ = svc.send(cid, f"m{i}")
    assert not any(c[0] == SUMMARY_SYSTEM_PROMPT for c in llm.calls)
    assert svc.get(cid).summary == ""


def test_summary_failure_falls_back_to_truncation():
    llm, svc = make_service(max_recent=4, summarize_batch=1)
    cid = None
    for i in range(3):
        cid, _ = svc.send(cid, f"m{i}")
    # Next send triggers compression; make the summary call fail.
    llm.fail_next = LLMError("boom", status=503)
    cid, reply = svc.send(cid, "m3")                # summary call fails -> truncation, chat call works
    conv = svc.get(cid)
    assert reply.startswith("reply-")
    assert conv.summary == ""
    assert len(conv.messages) <= 6


def test_failed_call_rolls_back_user_message():
    llm, svc = make_service()
    cid, _ = svc.send(None, "first")
    before = list(svc.get(cid).messages)
    llm.fail_next = LLMError("down", status=503)
    with pytest.raises(LLMError):
        svc.send(cid, "this one fails")
    assert svc.get(cid).messages == before           # nothing dangling, user can retry


def test_conversations_are_isolated():
    llm, svc = make_service()
    a, _ = svc.send(None, "I am A")
    b, _ = svc.send(None, "I am B")
    assert a != b
    assert all("I am B" not in m["content"] for m in svc.get(a).messages)


# ------------------------------------------------------------------ HTTP layer
@pytest.fixture
def client():
    llm, svc = make_service()
    app = create_app(svc)
    return llm, app.test_client()


def test_api_chat_happy_path(client):
    _, c = client
    r = c.post("/api/chat", json={"message": "hello"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["reply"].startswith("reply-") and body["conversation_id"]
    r2 = c.post("/api/chat", json={"message": "again", "conversation_id": body["conversation_id"]})
    assert r2.get_json()["context"]["messages_in_window"] == 4


def test_api_rejects_empty_and_oversized(client):
    _, c = client
    assert c.post("/api/chat", json={"message": "   "}).status_code == 400
    assert c.post("/api/chat", json={"message": "x" * 5000}).status_code == 400
    assert c.post("/api/chat", data="not json").status_code == 400


@pytest.mark.parametrize("status", [429, 503, 504])
def test_api_shows_clear_message_on_llm_failure(client, status):
    llm, c = client
    llm.fail_next = LLMError("Friendly failure message", status=status)
    r = c.post("/api/chat", json={"message": "hi"})
    assert r.status_code == status
    assert r.get_json()["error"] == "Friendly failure message"


def test_api_unexpected_error_is_hidden_from_user(client):
    llm, c = client
    llm.fail_next = RuntimeError("secret internal detail")
    r = c.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 500
    assert "secret" not in r.get_json()["error"]
