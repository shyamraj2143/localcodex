import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agent.ollama import OllamaClient


class OllamaClientTests(unittest.TestCase):
    def setUp(self):
        self.tools = [{
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a file with complete contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        }]
        self.messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Create hello.py"},
        ]

    def test_local_default_uses_json_protocol_instead_of_native_tools(self):
        client = OllamaClient(model="test-model")
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"name":"write_file","arguments":{"path":"hello.py","content":"print(1)"}}',
                tool_calls=None,
            ))]
        )
        client.client = Mock()
        client.client.chat.completions.create.return_value = completion

        result = client.chat(self.messages, tools=self.tools)

        kwargs = client.client.chat.completions.create.call_args.kwargs
        self.assertIs(result, completion)
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)
        instruction = kwargs["messages"][-1]["content"]
        self.assertIn("write_file", instruction)
        self.assertIn('"content"', instruction)
        self.assertIn("exactly ONE valid JSON object", instruction)

    def test_native_tool_mode_can_be_enabled_explicitly(self):
        client = OllamaClient(model="test-model", force_json_tools=False)
        completion = SimpleNamespace(choices=[])
        client.client = Mock()
        client.client.chat.completions.create.return_value = completion

        client.chat(self.messages, tools=self.tools)

        kwargs = client.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["tools"], self.tools)
        self.assertEqual(kwargs["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()
