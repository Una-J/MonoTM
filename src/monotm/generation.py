"""Generation settings and conservative checks for Chat Completions endpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import re
from urllib.parse import urlparse


REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
TOKEN_PARAMETERS = ("max_tokens", "max_completion_tokens")


@dataclass
class GenerationConfig:
    temperature: float | None = 0.0
    token_limit: int = 1024
    token_limit_parameter: str = "max_tokens"
    reasoning_effort: str | None = None

    def validate(self):
        if self.temperature is not None and (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError("temperature must be a finite number between 0 and 2, or null to omit it")
        if isinstance(self.token_limit, bool) or not isinstance(self.token_limit, int) or self.token_limit < 1:
            raise ValueError("token_limit must be a positive integer")
        if self.token_limit_parameter not in TOKEN_PARAMETERS:
            raise ValueError("token_limit_parameter must be max_tokens or max_completion_tokens")
        if self.reasoning_effort is not None and self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("reasoning_effort must be null or one of: " + ", ".join(REASONING_EFFORTS))

    def to_dict(self):
        return asdict(self)

    def request_parameters(self):
        self.validate()
        values = {self.token_limit_parameter: self.token_limit}
        if self.temperature is not None:
            values["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            values["reasoning_effort"] = self.reasoning_effort
        return values

    @classmethod
    def from_dict(cls, values, *, base=None):
        if not isinstance(values, dict):
            raise ValueError("Generation settings must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError("Unknown generation settings: " + ", ".join(sorted(unknown)))
        merged = (base or cls()).to_dict()
        merged.update(values)
        result = cls(**merged)
        result.validate()
        return result


def compatibility_issues(base_url, model, generation, role):
    """Check documented model IDs only on their provider's official endpoint.

    Aliases, proxies and unrecognized models are checked by the provider when
    a request is made. No values are changed by this check.
    """
    host = urlparse(base_url).hostname
    model_id = re.sub(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$", "", model.lower())
    issues = []

    def issue(message, setting, option, value):
        cli_value = ("omit" if setting == "reasoning_effort" else "none") if value == "null" else value
        issues.append(f"{role.title()} ({model}): {message} Set {role}_generation.{setting} "
                      f"to {value} (--{role}-{option} {cli_value}).")

    if host == "api.openai.com":
        o_series = re.fullmatch(r"o1(?:-mini|-preview)?|o3(?:-mini)?|o4-mini", model_id) is not None
        gpt5 = model_id in ("gpt-5", "gpt-5-mini", "gpt-5-nano")
        gpt5_switchable = model_id in ("gpt-5.1", "gpt-5.2")
        if (o_series or gpt5 or gpt5_switchable) and generation.token_limit_parameter == "max_tokens":
            issue("Use max_completion_tokens for this model.", "token_limit_parameter",
                  "token-limit-parameter", "max_completion_tokens")
        if ((o_series and generation.temperature not in (None, 1))
                or (gpt5 and generation.temperature is not None)
                or (gpt5_switchable and generation.reasoning_effort not in (None, "none")
                    and generation.temperature is not None)):
            issue("This temperature/reasoning combination is not supported.", "temperature", "temperature", "null")
        if (o_series or gpt5) and generation.reasoning_effort == "none":
            issue("This model does not support reasoning_effort=none.", "reasoning_effort", "reasoning-effort", "null")
    elif host == "api.anthropic.com":
        restricted = model_id in {"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-opus-5-5",
                                  "claude-sonnet-5", "claude-sonnet-5-5", "claude-haiku-5-5",
                                  "claude-fable-5", "claude-fable-5-1", "claude-mythos-5",
                                  "claude-mythos-5-1", "claude-mythos-preview"}
        if restricted and generation.temperature not in (None, 1):
            issue("This model does not support a non-default temperature.", "temperature", "temperature", "null")
    return issues
