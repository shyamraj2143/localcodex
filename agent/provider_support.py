"""Shared bounded retry and safe error formatting for provider clients."""

import os
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError

from agent.orchestration import redact_secrets


def credential_present(name):
    value = os.getenv(name, "").strip()
    return bool(value and not value.lower().startswith(("your_", "replace", "put_", "${", "<")))


def status_code(error):
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"\b(?:HTTP\s*)?(\d{3})\b", str(error), re.IGNORECASE)
    return int(match.group(1)) if match else None


def is_transient(error):
    if isinstance(error, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError,
                          TimeoutError, ConnectionError)):
        return True
    code = status_code(error)
    return code in {408, 409, 425, 429} or code is not None and code >= 500


def is_auth_error(error):
    code = status_code(error)
    return code in {401, 403} or "invalid_api_key" in str(error).lower()


def safe_error(error, key_names=()):
    message = redact_secrets(str(error))
    for key_name in key_names:
        secret = os.getenv(key_name, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:1200]


def retry_delay(error, attempt, base_delay):
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    try:
        retry_after = float(headers.get("retry-after", 0))
    except (TypeError, ValueError, AttributeError):
        retry_after = 0
    delay = retry_after if retry_after > 0 else float(base_delay) * (2 ** attempt)
    return max(0.1, min(delay, 30.0))


def provider_error(provider, model, error):
    code = status_code(error)
    if is_auth_error(error):
        status = f" (HTTP {code})" if code else ""
        detail = safe_error(error, ("GROQ_API_KEY", "NVIDIA_API_KEY"))
        message = (
            f"{provider} authentication/authorization failed{status}. "
            f"Check the configured API key and model access. Provider detail: {detail}"
        )
    else:
        status = f" (HTTP {code})" if code else ""
        message = f"{provider} request for model '{model}' failed{status}: {safe_error(error, ('GROQ_API_KEY', 'NVIDIA_API_KEY'))}"
    return RuntimeError(message)
