import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.nvidia import NvidiaClient


class FakeCompletions:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.error, Exception):
            raise self.error
        if self.error and "tools" in kwargs:
            raise Exception(self.error)
        return SimpleNamespace(choices=[], ok=True)


class NvidiaClientTests(unittest.TestCase):
    def test_native_tool_calling_request_includes_tools(self):
        client = NvidiaClient.__new__(NvidiaClient)
        client.model = "poolside/laguna-xs-2.1"
        client.temperature = 0.1
        client.max_tokens = 100
        client.max_attempts = 1
        client.base_delay = 0.1
        completions = FakeCompletions()
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        tools = [{"type": "function", "function": {"name": "list_files"}}]

        result = client.chat([{"role": "user", "content": "Inspect workspace"}], tools)

        self.assertTrue(result.ok)
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(completions.calls[0]["tools"], tools)

    def test_unsupported_native_tools_fall_back_to_json_instruction(self):
        client = NvidiaClient.__new__(NvidiaClient)
        client.model = "poolside/laguna-xs-2.1"
        client.temperature = 0.1
        client.max_tokens = 100
        client.max_attempts = 1
        client.base_delay = 0.1
        completions = FakeCompletions("This model does not support tools")
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        messages = [{"role": "user", "content": "Create a GUI"}]
        tools = [{"type": "function", "function": {"name": "write_file"}}]

        result = client.chat(messages, tools)

        self.assertTrue(result.ok)
        self.assertEqual(len(completions.calls), 2)
        self.assertIn("tools", completions.calls[0])
        self.assertNotIn("tools", completions.calls[1])
        self.assertIn("write_file", completions.calls[1]["messages"][-1]["content"])
        self.assertEqual(len(messages), 1)

    def test_rate_limit_retries_but_invalid_model_is_not_retried(self):
        class APIError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={})

        class SequenceCompletions:
            def __init__(self, outcomes): self.outcomes, self.calls = list(outcomes), []
            def create(self, **kwargs):
                self.calls.append(kwargs)
                outcome = self.outcomes.pop(0)
                if isinstance(outcome, Exception): raise outcome
                return outcome

        client = NvidiaClient.__new__(NvidiaClient)
        client.model, client.temperature, client.max_tokens = "qwen/qwen2.5-coder-32b-instruct", 0.1, 100
        client.max_attempts, client.base_delay = 3, 0.1
        response = SimpleNamespace(choices=[])
        completions = SequenceCompletions([APIError("rate limited"), response])
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("agent.nvidia.time.sleep") as sleep:
            self.assertIs(client.chat([{"role": "user", "content": "Reply OK"}]), response)
        self.assertEqual(len(completions.calls), 2)
        sleep.assert_called_once()

        class BadRequest(Exception):
            status_code = 400
        completions = SequenceCompletions([BadRequest("unknown model"), response])
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("agent.nvidia.time.sleep") as sleep, self.assertRaisesRegex(RuntimeError, "HTTP 400"):
            client.chat([{"role": "user", "content": "Reply OK"}])
        self.assertEqual(len(completions.calls), 1)
        sleep.assert_not_called()

    def test_retry_callback_reports_transient_failure_without_printing(self):
        class APIError(Exception):
            status_code = 503

        client = NvidiaClient.__new__(NvidiaClient)
        client.model, client.temperature, client.max_tokens = "openai/gpt-oss-20b", 0.1, 100
        client.max_attempts, client.base_delay = 2, 0.1
        retries = []
        client.on_retry = lambda *args: retries.append(args)
        response = SimpleNamespace(choices=[])
        completions = FakeCompletions(error=APIError("service unavailable"))
        original_create = completions.create
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise APIError("service unavailable")
            return response
        completions.create = create
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("agent.nvidia.time.sleep"):
            self.assertIs(client.chat([{"role": "user", "content": "Reply OK"}]), response)
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0][0:2], (1, 1))

    def test_retry_budget_is_exhausted_without_an_extra_attempt(self):
        class APIError(Exception):
            status_code = 503

        class AlwaysFail:
            def __init__(self): self.calls = 0
            def create(self, **kwargs):
                self.calls += 1
                raise APIError("service unavailable")

        client = NvidiaClient.__new__(NvidiaClient)
        client.model, client.temperature, client.max_tokens = "openai/gpt-oss-20b", 0.1, 100
        client.max_attempts, client.base_delay = 2, 0.1
        client.on_retry = lambda *args: None
        completions = AlwaysFail()
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("agent.nvidia.time.sleep") as sleep, self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            client.chat([{"role": "user", "content": "Reply OK"}])
        self.assertEqual(completions.calls, 2)
        sleep.assert_called_once()

    def test_authentication_failure_is_not_retried(self):
        class Unauthorized(Exception):
            status_code = 401

        completions = FakeCompletions(error=Unauthorized("invalid_api_key"))
        client = NvidiaClient.__new__(NvidiaClient)
        client.model, client.temperature, client.max_tokens = "openai/gpt-oss-20b", 0.1, 100
        client.max_attempts, client.base_delay = 3, 0.1
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("agent.nvidia.time.sleep") as sleep, self.assertRaisesRegex(RuntimeError, "authentication/authorization failed"):
            client.chat([{"role": "user", "content": "Reply OK"}])
        self.assertEqual(len(completions.calls), 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
