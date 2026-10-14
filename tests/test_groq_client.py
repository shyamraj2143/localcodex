from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.groq import GroqClient


class StatusError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers={})


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_with(outcomes, attempts=3):
    client = GroqClient.__new__(GroqClient)
    client.model = "qwen/qwen3.8-27b"
    client.temperature = 0.1
    client.max_tokens = 64
    client.max_attempts = attempts
    client.base_delay = 0.1
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(outcomes)))
    return client


def test_groq_retries_rate_limit_then_returns_normalized_openai_response():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    client = client_with([StatusError(429, "rate limit"), response])
    with patch("agent.groq.time.sleep") as sleep:
        actual = client.chat([{"role": "user", "content": "Reply OK"}])
    assert actual is response
    assert len(client.client.chat.completions.calls) == 2
    sleep.assert_called_once()


def test_groq_does_not_retry_invalid_model_or_authentication_errors():
    for error, expected in ((StatusError(400, "model is not available"), "HTTP 400"),
                            (StatusError(401, "invalid_api_key"), "authentication/authorization failed")):
        client = client_with([error, SimpleNamespace()], attempts=4)
        with patch("agent.groq.time.sleep") as sleep, pytest.raises(RuntimeError, match=expected):
            client.chat([{"role": "user", "content": "Reply OK"}])
        assert len(client.client.chat.completions.calls) == 1
        sleep.assert_not_called()
