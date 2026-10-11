"""Small Ollama JSON client with per-instance usage counters and safe errors."""

import json
import math
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.config import LLMConfig


class LLMError(RuntimeError):
    """Provider unavailable or response malformed; callers can use a fallback."""


@dataclass
class Usage:
    """Counters owned by one client, suitable for a single workflow run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0


class LLMClient:
    """Use the Ollama chat protocol locally or on an authenticated hosted server."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self.usage = Usage()

    @property
    def mock(self) -> bool:
        """Expose offline mode without letting callers depend on provider details."""
        return self.config.mock

    def complete_json(self, system: str, prompt: str) -> dict:
        """Return a JSON object or a sanitized typed error; never log credentials."""
        if self.mock:
            raise LLMError("Mock mode requires deterministic parsing or writing")
        cfg = self.config
        url = urlsplit(cfg.base_url)
        if cfg.provider not in {"ollama", "hosted"}:
            raise LLMError("Unsupported provider; use ollama or hosted")
        if url.scheme not in {"http", "https"} or url.username or url.password:
            raise LLMError("Invalid provider URL")
        if cfg.provider == "hosted" and (url.scheme != "https" or not cfg.api_key):
            raise LLMError("Hosted mode requires HTTPS and LLM_API_KEY")
        if not math.isfinite(cfg.timeout) or cfg.timeout <= 0:
            raise LLMError("Timeout must be positive and finite")
        payload = {"model": cfg.model, "stream": False, "format": "json",
                   "options": {"temperature": 0}, "messages": [
                       {"role": "system", "content": system},
                       {"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = "Bearer " + cfg.api_key
        request = Request(cfg.base_url.rstrip("/") + "/api/chat",
                          json.dumps(payload).encode(), headers)
        self.usage.calls += 1
        try:
            with urlopen(request, timeout=cfg.timeout) as response:
                body = json.load(response)
            self.usage.input_tokens += max(0, int(body.get("prompt_eval_count", 0)))
            self.usage.output_tokens += max(0, int(body.get("eval_count", 0)))
            result = json.loads(body["message"]["content"])
            if not isinstance(result, dict):
                raise ValueError("Expected JSON object")
            return result
        except (URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, AttributeError):
            # Provider bodies can contain credentials; do not include them in errors.
            self.usage.failures += 1
            raise LLMError("LLM request failed or returned invalid JSON") from None
