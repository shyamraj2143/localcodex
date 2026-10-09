import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from agent.loop import (
    MultiModelAgent,
    _expected_artifact_paths,
    _is_tool_result_json,
    recover_text_tool_calls,
)
from tools.files import write_file
from tools.registry import create_folder
from tools.terminal import run_command


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages_seen = []
        self.tools_seen = []

    def chat(self, messages, tools=None):
        self.messages_seen.append(list(messages))
        self.tools_seen.append(tools)
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

    def test_detects_echoed_tool_result_json(self):
        self.assertTrue(_is_tool_result_json(
            '{"success":true,"message":"Folder created","path":"gui"}'
        ))
        self.assertFalse(_is_tool_result_json(
            '{"name":"create_folder","arguments":{"path":"gui"}}'
        ))

    def test_expected_artifacts_include_file_inside_named_folder(self):
        paths = _expected_artifact_paths(
            "Create a folder named calculator_gui. Inside it, create main.py containing code."
        )
        self.assertEqual(paths, ["calculator_gui", "calculator_gui/main.py"])

    def test_integrate_recovers_echo_and_continues_to_create_file(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake_client = FakeClient([
                '{"name":"create_folder","arguments":{"path":"calculator_gui"}}',
                '{"success":true,"message":"Folder created successfully.","path":"calculator_gui","absolute_path":"calculator_gui"}',
                json.dumps({
                    "name": "write_file",
                    "arguments": {
                        "path": "calculator_gui/main.py",
                        "content": "print('calculator ready')\n",
                    },
                }),
                '{"name":"run_command","arguments":{"command":"python -m pip --version"}}',
                "Created and verified calculator_gui/main.py.",
            ])

            agent = MultiModelAgent(
                config={
                    "provider": "ollama",
                    "local_model": "test-model",
                    "max_agent_steps": 8,
                },
                tools={
                    "create_folder": create_folder,
                    "write_file": write_file,
                    "run_command": run_command,
                },
                tool_schemas=[],
                workspace=workspace,
            )
            agent.get_client = lambda role="coder": fake_client

            result = agent.integrate(
                "Create a folder named calculator_gui. Inside it, create main.py containing code.",
                [],
            )

            target = os.path.join(workspace, "calculator_gui", "main.py")
            self.assertTrue(os.path.isfile(target))
            with open(target, "r", encoding="utf-8") as file:
                self.assertEqual(file.read(), "print('calculator ready')\n")
            with open(target, "r", encoding="utf-8") as file:
                compile(file.read(), target, "exec")
            self.assertEqual(result, "Created and verified calculator_gui/main.py.")
            self.assertEqual(len(fake_client.messages_seen), 5)
            # The echoed JSON response is treated as an intermediate result, not final output.
            self.assertIn(
                "RESULT of tool create_folder",
                fake_client.messages_seen[1][-1]["content"],
            )
            self.assertEqual(fake_client.messages_seen[1][-1]["role"], "user")
            self.assertEqual(fake_client.messages_seen[1][-2]["role"], "assistant")
            self.assertNotIn("tool_calls", fake_client.messages_seen[1][-2])


    def test_empty_response_retries_without_tool_schemas(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake_client = FakeClient([
                "",
                json.dumps({
                    "name": "create_folder",
                    "arguments": {"path": "gui"},
                }),
                json.dumps({
                    "name": "write_file",
                    "arguments": {
                        "path": "gui/main.py",
                        "content": "print('gui ready')\\n",
                    },
                }),
                '{"name":"run_command","arguments":{"command":"python -m pip --version"}}',
                "Created and verified gui/main.py.",
            ])
            agent = MultiModelAgent(
                config={
                    "provider": "ollama",
                    "local_model": "test-model",
                    "max_agent_steps": 8,
                },
                tools={
                    "create_folder": create_folder,
                    "write_file": write_file,
                    "run_command": run_command,
                },
                tool_schemas=[{"type": "function", "function": {"name": "write_file"}}],
                workspace=workspace,
            )
            agent.get_client = lambda role="coder": fake_client

            result = agent.integrate(
                "Create a folder named gui. Inside it, create main.py containing code.",
                [],
            )

            target = os.path.join(workspace, "gui", "main.py")
            self.assertTrue(os.path.isfile(target))
            self.assertEqual(result, "Created and verified gui/main.py.")
            self.assertIsNone(fake_client.tools_seen[1])
            self.assertIsNotNone(fake_client.tools_seen[0])

    def test_premature_done_is_retried_when_requested_file_is_missing(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake_client = FakeClient([
                "Task completed.",
                json.dumps({
                    "name": "write_file",
                    "arguments": {
                        "path": "calculator_gui/main.py",
                        "content": "print('hello')\n",
                    },
                }),
                '{"name":"run_command","arguments":{"command":"python -m pip --version"}}',
                "Created calculator_gui/main.py and verified it.",
            ])
            agent = MultiModelAgent(
                config={
                    "provider": "ollama",
                    "local_model": "test-model",
                    "max_agent_steps": 8,
                },
                tools={"write_file": write_file, "run_command": run_command},
                tool_schemas=[],
                workspace=workspace,
            )
            agent.get_client = lambda role="coder": fake_client
            result = agent.integrate(
                "Create a folder named calculator_gui. Inside it, create main.py containing code.",
                [],
            )
            self.assertTrue(
                os.path.isfile(os.path.join(workspace, "calculator_gui", "main.py"))
            )
            self.assertEqual(result, "Created calculator_gui/main.py and verified it.")
            self.assertEqual(len(fake_client.messages_seen), 4)


    def test_recovers_multiple_json_objects_and_deduplicates_identical_calls(self):
        raw = (
            'Here is an action: {"name":"create_folder","arguments":{"path":"gui"}} '
            'and the same action again: {"name":"create_folder","arguments":{"path":"gui"}}'
        )
        calls = recover_text_tool_calls(raw, {"create_folder": lambda **kwargs: None})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].function.name, "create_folder")

    def test_repeated_successful_folder_creation_does_not_block_file_creation(self):
        with tempfile.TemporaryDirectory() as workspace:
            fake_client = FakeClient([
                '{"name":"create_folder","arguments":{"path":"gui"}}',
                '{"name":"create_folder","arguments":{"path":"gui"}}',
                json.dumps({
                    "name": "write_file",
                    "arguments": {
                        "path": "gui/main.py",
                        "content": "print('created after duplicate')\\n",
                    },
                }),
                '{"name":"run_command","arguments":{"command":"python -m pip --version"}}',
                "Created and verified gui/main.py.",
            ])
            agent = MultiModelAgent(
                config={"provider": "ollama", "local_model": "test-model", "max_agent_steps": 8},
                tools={"create_folder": create_folder, "write_file": write_file, "run_command": run_command},
                tool_schemas=[],
                workspace=workspace,
            )
            agent.get_client = lambda role="coder": fake_client
            result = agent.integrate(
                "Create a folder named gui. Inside it, create main.py containing code.",
                [],
            )
            target = os.path.join(workspace, "gui", "main.py")
            self.assertTrue(os.path.isfile(target))
            with open(target, "r", encoding="utf-8") as file:
                self.assertEqual(file.read(), "print('created after duplicate')\\n")
            self.assertEqual(result, "Created and verified gui/main.py.")
            self.assertEqual(len(fake_client.messages_seen), 5)

if __name__ == "__main__":
    unittest.main()
