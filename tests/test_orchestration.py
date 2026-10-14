import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import MultiModelAgent
from agent.orchestration import (
    SpecificationError,
    completeness_issues,
    fallback_specification,
    parse_specification,
    redact_secrets,
    validate_specification,
)
from tools.files import read_file, write_file
from tools.terminal import run_command


class OrchestrationTests(unittest.TestCase):
    def test_game_fallback_spec_is_complete_and_dynamic(self):
        spec = fallback_specification("Build a playable game called comet_run")
        self.assertEqual(spec["task_type"], "game_development")
        self.assertEqual(spec["project_name"], "comet_run")
        self.assertIn("tests/test_game_logic.py", [item["path"] for item in spec["expected_files"]])

    def test_structured_spec_parses_json_and_rejects_unsafe_paths(self):
        source = fallback_specification("Fix the app")
        self.assertEqual(parse_specification("plan: " + json.dumps(source)), source)
        source["expected_files"] = [{"path": "../outside.py", "responsibility": "escape"}]
        with self.assertRaises(SpecificationError):
            validate_specification(source)

    def test_game_completeness_rejects_hello_world_and_accepts_game_logic(self):
        spec = {"task_type": "game_development", "expected_files": [{"path": "main.py", "responsibility": "game"}]}
        with tempfile.TemporaryDirectory() as workspace:
            Path(workspace, "main.py").write_text("print('Hello World')\n", encoding="utf-8")
            self.assertTrue(any("trivial" in issue for issue in completeness_issues(workspace, spec)))
            code = """
import pygame
def update_game(state):
    state['score'] += 1
    return state
def draw_game(screen, state):
    pygame.display.flip()
def game_loop():
    running = True
    score = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                score += 1
                if event.key == pygame.K_ESCAPE:
                    running = False
        draw_game(None, {'score': score})
    return score
"""
            Path(workspace, "main.py").write_text(code, encoding="utf-8")
            self.assertEqual(completeness_issues(workspace, spec), [])

    def test_simple_task_skips_hosted_architect(self):
        agent = MultiModelAgent({"provider": "ollama"}, {}, [], tempfile.gettempdir())
        agent.get_client = lambda *args, **kwargs: self.fail("simple task should not call a provider")
        spec, meta = agent._architect_task("What is 2 + 2?")
        self.assertEqual(meta["status"], "SKIPPED (simple task)")
        self.assertTrue(spec["intended_outcome"])

    def test_architect_uses_configured_model_and_falls_back_on_malformed_json(self):
        agent = MultiModelAgent({"provider": "ollama", "architect_provider": "groq", "models": {"planner": "configured-planner"}}, {}, [], tempfile.gettempdir())
        fake_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))])
        class FakeClient:
            model = "configured-planner"
            def chat(self, messages, tools=None):
                self.asserted_messages = messages
                return fake_response
        fake = FakeClient()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            agent.get_client = lambda *args, **kwargs: fake
            spec, meta = agent._architect_task("Build a game")
        self.assertEqual(meta["status"], "FAILED; deterministic specification used")
        self.assertEqual(spec["task_type"], "game_development")

    def test_router_only_selects_configured_models_with_credentials(self):
        agent = MultiModelAgent({"provider": "groq", "models": {"planner": "groq-planner"}, "nvidia_model": "nvidia-coder"}, {}, [], tempfile.gettempdir())
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-test", "NVIDIA_API_KEY": "nvidia-test"}):
            self.assertEqual(agent._hosted_candidates("planner", "auto"), ["groq"])
            self.assertEqual(agent._model_for("groq", "planner"), "groq-planner")
            self.assertEqual(agent._model_for("nvidia", "coder"), "nvidia-coder")

    def test_selected_local_provider_does_not_silently_route_to_hosted_provider(self):
        agent = MultiModelAgent({"provider": "ollama", "models": {"planner": "groq-planner"}, "nvidia_model": "nvidia-coder"}, {}, [], tempfile.gettempdir())
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-test", "NVIDIA_API_KEY": "nvidia-test"}):
            self.assertEqual(agent._hosted_candidates("planner", "auto"), [])

    def test_secret_redaction_and_sensitive_file_read_block(self):
        self.assertNotIn("supersecret123", redact_secrets("api_key=supersecret123"))
        for credential in ("gsk_" + "a" * 24, "nvapi-" + "b" * 24, "Bearer " + "c" * 24):
            self.assertNotIn(credential.split()[-1], redact_secrets(credential))
        called = []
        agent = MultiModelAgent({}, {"read_file": lambda *args, **kwargs: called.append(True)}, [], tempfile.gettempdir())
        result = agent.execute_tool("read_file", {"path": ".env"})
        self.assertFalse(result["success"])
        self.assertEqual(called, [])

    def test_architect_does_not_switch_provider_after_malformed_response(self):
        agent = MultiModelAgent({"provider": "ollama", "architect_provider": "groq", "models": {"planner": "groq-planner"}, "nvidia_model": "nvidia-model"}, {}, [], tempfile.gettempdir())
        valid = json.dumps(fallback_specification("Build a game"))
        seen = []
        class FakeClient:
            def __init__(self, model, content): self.model, self.content = model, content
            def chat(self, messages, tools=None):
                seen.append(messages[-1]["content"])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])
        clients = {"groq": FakeClient("groq-planner", "invalid"), "nvidia": FakeClient("nvidia-model", valid)}
        requested = []
        with patch.dict(os.environ, {"GROQ_API_KEY": "groq-test", "NVIDIA_API_KEY": "nvidia-test"}):
            agent.get_client = lambda role, provider=None, model=None: (requested.append(provider), clients[provider])[1]
            spec, meta = agent._architect_task("Build a game; api_key=supersecret987654")
        self.assertEqual(meta["provider"], "groq")
        self.assertEqual(requested, ["groq"])
        self.assertIn("FAILED", meta["status"])
        self.assertEqual(spec["task_type"], "game_development")
        self.assertTrue(seen)
        self.assertNotIn("supersecret987654", " ".join(seen))

    def test_local_partial_result_escalates_once_to_configured_hosted_coder(self):
        agent = MultiModelAgent({"provider": "ollama", "models": {"coder": "configured-groq-coder"}, "hosted_coder_escalation": True, "escalation_provider": "groq", "max_escalations": 1}, {}, [], tempfile.gettempdir())
        spec = fallback_specification("Fix the bug")
        agent._architect_task = lambda task, previous_report=None: (spec, {"provider": "none", "model": "none", "status": "SKIPPED (simple task)"})
        calls = []
        def fake_integrate(task, reports=None, coder_provider=None):
            calls.append((reports, coder_provider))
            if coder_provider is None:
                agent.last_task_report = {"task": task, "changed_files": [], "commands": [], "static_checks": [], "verification_status": "PARTIAL"}
                return "local attempt partial"
            agent.last_task_report = {"task": task, "changed_files": [], "commands": [], "static_checks": [], "verification_status": "PASS"}
            return "hosted repair passed"
        agent.integrate = fake_integrate
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "NVIDIA_API_KEY": ""}):
            result = agent.run("Fix the bug")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], "groq")
        self.assertIn("hosted repair passed", result)

    def test_hosted_repair_receives_changed_source_and_actual_command_failure(self):
        with tempfile.TemporaryDirectory() as workspace:
            write_file(workspace, "game/gui.py", "def run_gui():\n    raise RuntimeError('window failed')\n")
            agent = MultiModelAgent(
                {"provider": "ollama", "models": {"coder": "groq-coder"}, "hosted_coder_escalation": True, "escalation_provider": "groq", "max_escalations": 1},
                {"read_file": read_file}, [], workspace,
            )
            specification = fallback_specification("Fix the game GUI")
            agent._architect_task = lambda task, previous_report=None: (
                specification, {"provider": "none", "model": "none", "status": "SKIPPED"}
            )
            captured = {}

            def fake_integrate(task, reports=None, coder_provider=None):
                if coder_provider is None:
                    agent.last_task_report = {
                        "task": task, "changed_files": ["game/gui.py"],
                        "commands": [{"command": "python game/main.py", "return_code": 1,
                                      "success": False, "stdout": "", "stderr": "ModuleNotFoundError: pygame"}],
                        "static_checks": [], "verification_status": "PARTIAL",
                    }
                    return "local attempt failed"
                captured["repair_context"] = reports["repair_context"]
                agent.last_task_report = {"task": task, "changed_files": [], "commands": [], "static_checks": [], "verification_status": "PASS"}
                return "hosted repair passed"

            agent.integrate = fake_integrate
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "NVIDIA_API_KEY": ""}):
                result = agent.run("is game me GUI open nahi hua")

        self.assertIn("hosted repair passed", result)
        self.assertIn("ModuleNotFoundError: pygame", captured["repair_context"])
        self.assertIn("raise RuntimeError", captured["repair_context"])

    def test_empty_test_discovery_does_not_satisfy_specification(self):
        agent = MultiModelAgent({}, {}, [], tempfile.gettempdir())
        agent.last_task_report = {
            "specification": {"task_type": "software_change", "expected_files": [], "test_commands": ["python -m unittest"]},
            "commands": [{"command": "python -m unittest discover", "return_code": 0, "success": True, "stdout": "Ran 0 tests in 0.0s"}],
        }
        self.assertTrue(any("not confirmed" in issue for issue in agent._task_completeness_issues()))

    def test_hello_world_game_is_rejected_then_repaired_and_tested(self):
        spec = fallback_specification("Build a playable game")
        game_files = {
            "main.py": '''
def move_player(position, direction, width=800, height=600):
    x, y = position
    dx, dy = direction
    return max(0, min(width, x + dx)), max(0, min(height, y + dy))

def main():
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    player = [400, 300]
    score = 0
    running = True
    game_over = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game_over:
                    score, game_over = 0, False
                elif not game_over:
                    score += 1
                    if event.key == pygame.K_LEFT:
                        player[:] = move_player(player, (-12, 0))
                    elif event.key == pygame.K_RIGHT:
                        player[:] = move_player(player, (12, 0))
        screen.fill((12, 18, 35))
        pygame.draw.circle(screen, (40, 220, 255), player, 18)
        pygame.display.set_caption(f"Comet Run | Score: {score}")
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()
''',
            "README.md": "# Comet Run\nInstall pygame with `pip install -r requirements.txt`, then run `python main.py`. Use arrow keys to move, R to restart after game over, and Esc to quit.\n",
            "requirements.txt": "pygame>=2.5,<3\n",
            "tests/test_game_logic.py": "import unittest\nfrom main import move_player\n\nclass GameLogicTests(unittest.TestCase):\n    def test_player_move_is_bounded(self):\n        self.assertEqual(move_player((790, 10), (20, -30)), (800, 0))\n\nif __name__ == '__main__':\n    unittest.main()\n",
        }

        def action(name, arguments):
            return json.dumps({"name": name, "arguments": arguments})

        initial = [
            action("write_file", {"path": "main.py", "content": "print('Hello World')\n"}),
            action("write_file", {"path": "README.md", "content": "# game\n"}),
            action("write_file", {"path": "requirements.txt", "content": "pygame\n"}),
            action("write_file", {"path": "tests/test_game_logic.py", "content": "import unittest\nclass T(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n"}),
            action("run_command", {"command": f'"{sys.executable}" -m unittest discover -s tests -v'}),
            "The game is complete.",
            "The game is complete.",
            json.dumps({"files": game_files}),
        ]

        class FakeClient:
            def __init__(self, responses): self.responses = list(responses)
            def chat(self, messages, tools=None):
                content = self.responses.pop(0)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))])

        with tempfile.TemporaryDirectory() as workspace:
            fake = FakeClient(initial)
            agent = MultiModelAgent(
                {"provider": "ollama", "local_model": "mock", "ollama_force_json_tools": True, "max_agent_steps": 12},
                {"write_file": write_file, "run_command": run_command}, [], workspace,
            )
            agent.get_client = lambda role="coder": fake
            result = agent.integrate("Build a playable game", {"specification": spec})
            self.assertIn("Verification status: PASS", result)
            self.assertTrue(os.path.isfile(Path(workspace, "main.py")))
            self.assertNotIn("Hello World", Path(workspace, "main.py").read_text(encoding="utf-8"))
            self.assertTrue(any("Ran 1 test" in command["stdout"] + command["stderr"] for command in agent.last_task_report["commands"]))


if __name__ == "__main__":
    unittest.main()
