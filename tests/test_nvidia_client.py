import unittest
from types import SimpleNamespace

from agent.nvidia import NvidiaClient


class FakeCompletions:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
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


if __name__ == "__main__":
    unittest.main()
