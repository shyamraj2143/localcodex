import unittest
from types import SimpleNamespace

from agent.ollama import OllamaClient


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" in kwargs:
            raise Exception(
                "registry.ollama.ai/library/deepseek-coder:1.3b does not support tools"
            )
        return SimpleNamespace(ok=True, choices=[])


class OllamaToolFallbackTests(unittest.TestCase):
    def test_retries_without_native_tools_and_requests_json_tool_call(self):
        client = OllamaClient(model="deepseek-coder:1.3b", force_json_tools=False)
        completions = FakeCompletions()
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_folder",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        messages = [{"role": "user", "content": "Create a folder named demo."}]

        response = client.chat(messages, tools=tools)

        self.assertTrue(response.ok)
        self.assertEqual(len(completions.calls), 2)
        self.assertIn("tools", completions.calls[0])
        self.assertNotIn("tools", completions.calls[1])
        self.assertNotIn("tool_choice", completions.calls[1])
        self.assertIn(
            "create_folder",
            completions.calls[1]["messages"][-1]["content"],
        )
        # The caller's original conversation is not mutated.
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
