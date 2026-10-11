"""Small, provider-agnostic LLM client.

Responsibility: send one prompt to the configured LLM and return its text plus
token usage. Nothing else. The client knows nothing about plans, KPIs or the
workflow; the parser and writer decide what to ask.

Configuration comes ONLY from environment variables (normally loaded from
``.env``, which Git ignores):

    LLM_PROVIDER     ollama | openai | anthropic | gemini | mock   (default: ollama)
    LLM_MODEL        model name           (default: qwen2.5:7b-instruct)
    MOCK_LLM         true/false           (default: false; true forces mock mode)
    LLM_TIMEOUT_S    seconds per call     (default: 30)
    OLLAMA_BASE_URL  default http://localhost:11434
                     (inside Docker use http://host.docker.internal:11434)
    OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY   for hosted providers

Design decisions
----------------
* Standard-library HTTP (``urllib``) only, so no new dependency is added to the
  team's requirements.
* In mock mode the client refuses to be called (``LLMUnavailableError``). The
  parser and writer check ``is_mock`` first and go straight to their
  deterministic fallbacks, so mock mode needs no key, network or model.
* Secrets are never logged and never included in error messages.
* Each client instance counts its own calls and tokens. There is no global
  mutable state; the workflow can create one client per run.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from app.llm.models import LLMResponse

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "gemini", "mock")


class LLMError(Exception):
    """Base class for all LLM-module errors."""


class LLMUnavailableError(LLMError):
    """The LLM cannot be reached (mock mode, timeout, network error, missing key)."""


class LLMResponseError(LLMError):
    """The LLM answered, but the response was malformed or unusable."""


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a true/false environment variable in a forgiving way."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    """Immutable LLM configuration. Build it with ``LLMSettings.from_env()``."""

    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    mock: bool = False
    timeout_s: float = 30.0
    ollama_base_url: str = "http://localhost:11434"
    max_output_tokens: int = 800

    @classmethod
    def from_env(cls) -> "LLMSettings":
        """Create settings from environment variables (see module docstring)."""
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER={provider!r} is not supported; use one of {SUPPORTED_PROVIDERS}"
            )
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct").strip(),
            # Mock mode is on if MOCK_LLM is true OR the provider is literally "mock".
            mock=_env_bool("MOCK_LLM") or provider == "mock",
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", "30")),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        )


@dataclass
class LLMClient:
    """Calls the configured LLM provider and tracks usage for this instance."""

    settings: LLMSettings = field(default_factory=LLMSettings.from_env)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_mock(self) -> bool:
        """True when no real LLM should be called."""
        return self.settings.mock

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def describe(self) -> dict[str, Any]:
        """Safe description for /health and traces (contains no secrets)."""
        return {
            "provider": "mock" if self.is_mock else self.settings.provider,
            "model": self.settings.model,
            "mock_llm": self.is_mock,
        }

    # ------------------------------------------------------------------ public
    def complete(self, system: str, user: str, json_mode: bool = False) -> LLMResponse:
        """Send one system + user prompt and return the model's text.

        Raises:
            LLMUnavailableError: mock mode, timeout, connection problem, missing key.
            LLMResponseError: the provider returned something we cannot read.
        """
        if self.is_mock:
            raise LLMUnavailableError("MOCK_LLM is enabled; no LLM call is made.")

        provider = self.settings.provider
        handler = {
            "ollama": self._call_ollama,
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
            "gemini": self._call_gemini,
        }[provider]

        # Count the attempt even if it fails: the trace should show real calls.
        self.calls += 1
        response = handler(system, user, json_mode)
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        logger.info(
            "llm_call provider=%s model=%s tokens=%d",
            provider, self.settings.model, response.total_tokens,
        )
        return response

    # ------------------------------------------------------------ HTTP helper
    def _post_json(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        """POST JSON and return parsed JSON, mapping failures to typed errors.

        ``url`` may contain an API key (Gemini), so it is never logged or put in
        an exception message.
        """
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # 4xx/5xx from the provider. Report the status code only.
            raise LLMUnavailableError(
                f"{self.settings.provider} returned HTTP {exc.code}"
            ) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            raise LLMUnavailableError(
                f"{self.settings.provider} unreachable or timed out "
                f"(timeout={self.settings.timeout_s}s)"
            ) from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Provider response was not valid JSON") from exc

    def _require_key(self, env_name: str) -> str:
        key = os.getenv(env_name, "").strip()
        if not key:
            raise LLMUnavailableError(f"{env_name} is not set")
        return key

    # -------------------------------------------------------------- providers
    def _call_ollama(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # temperature 0: we want the same parse for the same request.
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"  # Ollama constrains output to valid JSON
        data = self._post_json(f"{self.settings.ollama_base_url}/api/chat", payload, {})
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError("Unexpected Ollama response shape") from exc
        return LLMResponse(
            text=text,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    def _call_openai(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        key = self._require_key("OPENAI_API_KEY")
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {key}"},
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("Unexpected OpenAI response shape") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    def _call_anthropic(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        key = self._require_key("ANTHROPIC_API_KEY")
        if json_mode:
            # Anthropic has no JSON switch here; the instruction is enough and
            # the parser validates the result anyway.
            user = user + "\n\nRespond with a single JSON object only."
        payload = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_output_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        try:
            text = "".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LLMResponseError("Unexpected Anthropic response shape") from exc
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
        )

    def _call_gemini(self, system: str, user: str, json_mode: bool) -> LLMResponse:
        key = self._require_key("GEMINI_API_KEY")
        config: dict[str, Any] = {"temperature": 0}
        if json_mode:
            config["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.model}:generateContent?key={key}"
        )
        data = self._post_json(url, payload, {})
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("Unexpected Gemini response shape") from exc
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
        )


def get_client(settings: Optional[LLMSettings] = None) -> LLMClient:
    """Convenience factory: a fresh client using env settings unless given."""
    return LLMClient(settings=settings or LLMSettings.from_env())
