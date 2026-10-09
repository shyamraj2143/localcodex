import json
import unittest
from types import SimpleNamespace

from agent.loop import MultiModelAgent, recover_text_tool_calls


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages_seen = []

    def chat(self, messages, tools=None):
        self.messages_seen.append(list(messages))
        content = self.responses.pop(0)
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class RecoverTextToolCallTests(unittest.TestCase):
    def test_recovers_json_tool_call(self):
        raw = json.dumps({
            "name": "create_folder",
            "arguments": {"path": "calculator_gui"},
        })

        calls = recover_text_tool_calls(
            raw,
            {"create_folder": lambda **kwargs: None},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "create_folder")
        self.assertEqual(
            json.loads(calls[0].function.arguments),
            {"path": "calculator_gui"},
        )

    def test_ignores_unregistered_tool(self):
        calls = recover_text_tool_calls(
            '{"name":"delete_everything","arguments":{}}',
            {"create_folder": lambda **kwargs: None},
        )
        self.assertEqual(calls, [])

    def test_integrate_executes_recovered_json_tool_call(self):
        executed = []

        def create_folder(workspace, path):
            executed.append((workspace, path))
            return {"success": True, "path": path}

        fake_client = FakeClient([
            '{"name":"create_folder","arguments":{"path":"calculator_gui"}}',
            "Folder created and verified.",
        ])

        agent = MultiModelAgent(
            config={
                "models": {"coder": "test-model"},
                "max_agent_steps": 3,
            },
            tools={"create_folder": create_folder},
            tool_schemas=[],
            workspace="test-workspace",
        )
        agent.get_client = lambda role: fake_client

        result = agent.integrate("Create a calculator folder", [])

        self.assertEqual(executed, [("test-workspace", "calculator_gui")])
        self.assertEqual(result, "Folder created and verified.")
        self.assertEqual(len(fake_client.messages_seen), 2)
        self.assertEqual(fake_client.messages_seen[1][-1]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
