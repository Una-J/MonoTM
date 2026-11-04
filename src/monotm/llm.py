"""OpenAI-compatible HTTP client with response caching and attempt history."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from urllib import error, request, parse
from uuid import uuid4


class LLMError(RuntimeError):
    pass


def validate_base_url(base_url):
    try:
        url = parse.urlparse(base_url)
        valid = (url.scheme in ("https", "http") and url.netloc and not
                 (url.username or url.password or url.query or url.fragment))
        url.port  # Validate a supplied port without connecting.
    except (ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("base_url must be an HTTP(S) base URL without credentials, query parameters or fragments")
    return base_url.rstrip("/")


class ChatClient:
    def __init__(self, *, base_url, api_key, cache_dir, timeout=180, retries=3, max_tokens=1024):
        self.base_url = validate_base_url(base_url)
        if not api_key or not api_key.strip():
            raise ValueError("Set DEEPINFRA_API_KEY (or MONOTM_API_KEY) before calling the LLM")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.api_key = api_key.strip()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries = timeout, retries
        self.max_tokens = max_tokens
        self.attempt_dir = self.cache_dir / "attempts"
        self.attempt_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0

    def complete(self, prompt, model, *, validator=None, context=None, refresh=False):
        """Cache only accepted replies; save every attempt, including malformed ones.

        context identifies the feature, stage and validation document. refresh
        bypasses accepted cache entries when a response needs revision.
        """
        from .data import write_json
        payload = {"model": model, "temperature": 0.0, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        key = hashlib.sha256(json.dumps({"base_url": self.base_url, "payload": payload}, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        def log_attempt(**details):
            entry = {"attempt_id": uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
                     "request_key": key, "base_url": self.base_url, "request": payload,
                     "context": context or {}, **details}
            # Never persist the credential, even if an endpoint echoes it.
            write_json(self.attempt_dir / f"{entry['attempt_id']}.json", self._redact(entry))

        if path.exists() and not refresh:
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                text = cached["response"]
                if isinstance(text, str) and text.strip() and (validator is None or validator(text)):
                    with self._lock:
                        self.cache_hits += 1
                    log_attempt(status="cache_hit", response=text, raw_response=None,
                                finish_reason=cached.get("finish_reason"))
                    return text
            except (ValueError, KeyError):
                pass
        for attempt in range(self.retries):
            req = request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                  headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}, method="POST")
            raw, text, finish_reason, http_status = None, None, None, None
            fatal = False
            try:
                with self._lock:
                    self.calls += 1
                with request.urlopen(req, timeout=self.timeout) as response:
                    http_status = getattr(response, "status", 200)
                    raw = response.read().decode("utf-8", errors="replace")
                    result = json.loads(raw)
                finish_reason = result["choices"][0].get("finish_reason")
                text = result["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip() or (validator is not None and not validator(text)):
                    raise LLMError("LLM returned an empty or unparseable response")
                log_attempt(status="accepted", attempt=attempt + 1, response=text, raw_response=raw,
                            http_status=http_status, finish_reason=finish_reason)
                cached = {"base_url": self.base_url, "request": payload, "response": text,
                          "finish_reason": finish_reason}
                write_json(path, self._redact(cached))
                return text
            except error.HTTPError as exc:
                http_status = exc.code
                try:
                    if exc.fp is not None:
                        raw = exc.read().decode("utf-8", errors="replace")
                except (OSError, AttributeError):
                    pass
                fatal = exc.code not in (408, 429, 500, 502, 503, 504)
                problem = f"LLM request failed with HTTP {exc.code}"
            except (error.URLError, TimeoutError, ConnectionError):
                problem = "LLM request timed out or could not reach the endpoint"
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, LLMError):
                problem = "LLM returned an empty or unparseable response"
            log_attempt(status="failed", attempt=attempt + 1, response=text, raw_response=raw,
                        http_status=http_status, finish_reason=finish_reason, error=problem)
            if fatal:
                raise LLMError(f"{problem}; check the endpoint, model, and API key") from None
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"{problem} after {self.retries} attempts; inspect cache/llm/attempts for raw replies and prompts")

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self.api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value
