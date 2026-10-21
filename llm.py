"""Thin wrapper around the Gemini API with timeouts, retries and friendly errors."""
import logging
import time

import httpx
from google import genai
from google.genai import errors, types

log = logging.getLogger(__name__)


class LLMError(Exception):
    """An error whose message is safe to show directly to the end user."""

    def __init__(self, user_message: str, status: int = 502, retryable: bool = False):
        super().__init__(user_message)
        self.user_message = user_message
        self.status = status
        self.retryable = retryable


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # google-genai expects the timeout in milliseconds.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    # ------------------------------------------------------------------ public
    def generate(self, system_instruction: str, messages: list[dict]) -> str:
        """Send the conversation to the model and return the reply text.

        `messages` is a list of {"role": "user" | "assistant", "content": str}.
        """
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                return self._extract_text(response)
            except LLMError as err:
                if err.retryable and attempt < self.max_retries:
                    attempt += 1
                    self._backoff(attempt, err)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001 - translated below
                err = self._translate(exc)
                if err.retryable and attempt < self.max_retries:
                    attempt += 1
                    self._backoff(attempt, err)
                    continue
                raise err from exc

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _backoff(attempt: int, err: LLMError) -> None:
        delay = 2 ** (attempt - 1)  # 1s, 2s, 4s...
        log.warning("LLM call failed (%s); retry %d in %ss", err.user_message, attempt, delay)
        time.sleep(delay)

    @staticmethod
    def _extract_text(response) -> str:
        text = (getattr(response, "text", None) or "").strip()
        if text:
            return text
        # No text: the response was blocked or empty.
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMError(
                "I can't respond to that message. Try rephrasing it.", status=422
            )
        raise LLMError(
            "The model returned an empty response. Please try again.",
            status=502,
            retryable=True,
        )

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        """Map SDK / network exceptions to a user-friendly LLMError."""
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return LLMError(
                "The AI service took too long to respond. Please try again.",
                status=504,
                retryable=True,
            )
        if isinstance(exc, (httpx.TransportError, ConnectionError)):
            return LLMError(
                "Couldn't reach the AI service. Check your internet connection and try again.",
                status=503,
                retryable=True,
            )
        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            log.error("Gemini API error %s: %s", code, exc)
            if code in (401, 403):
                return LLMError(
                    "The server's API key was rejected. Please contact the site owner.",
                    status=500,
                )
            if code == 429:
                return LLMError(
                    "The AI service is rate-limited right now. Wait a few seconds and retry.",
                    status=429,
                    retryable=True,
                )
            if code is not None and code >= 500:
                return LLMError(
                    "The AI service is having trouble. Please try again shortly.",
                    status=503,
                    retryable=True,
                )
            return LLMError(
                "The AI service rejected the request. Please try again.", status=502
            )
        log.exception("Unexpected LLM failure", exc_info=exc)
        return LLMError("Something went wrong talking to the AI service.", status=502)
