import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import MultiModelAgent
from agent.ollama import OllamaClient
from agent.orchestration import fallback_specification
from tools.terminal import run_command


class OfflineModeTests(unittest.TestCase):
    def make_agent(self, extra=None):
        config = {"execution_mode": "offline", "provider": "groq", "local_model": "qwen2.5-coder:3b",
                  "hosted_coder_escalation": True, "task_architect_enabled": True}
        config.update(extra or {})
        return MultiModelAgent(config, {"run_command": run_command}, [], tempfile.gettempdir())

    def test_offline_blocks_all_hosted_candidates_and_direct_clients(self):
        agent = self.make_agent()
        with patch.dict("os.environ", {"GROQ_API_KEY": "test", "NVIDIA_API_KEY": "test"}):
            self.assertEqual(agent._hosted_candidates("planner", "auto"), [])
            self.assertEqual(agent._hosted_candidates("coder", "groq"), [])
        with self.assertRaisesRegex(RuntimeError, "Offline mode blocks"):
            agent.get_client("coder", provider="nvidia")
        self.assertEqual(agent.config["provider"], "ollama")

    def test_offline_uses_configured_local_model_and_never_escalates(self):
        agent = self.make_agent()
        client = agent.get_client("coder")
        self.assertEqual(client.model, "qwen2.5-coder:3b")
        self.assertEqual(agent._escalation_provider("ollama"), None)
        agent.get_client = lambda *args, **kwargs: self.fail("hosted client was requested")
        spec, metadata = agent._architect_task("Build a calculator")
        self.assertTrue(spec["expected_files"])
        self.assertEqual(metadata["provider"], "none")

    def test_offline_command_blocks_network_installs_and_allows_local_checks(self):
        with tempfile.TemporaryDirectory() as workspace:
            blocked = run_command(workspace, "python -m pip install pygame", offline=True)
            self.assertTrue(blocked["blocked"])
            self.assertIn("offline mode", blocked["error"])
            allowed = run_command(workspace, "python --version", offline=True)
            self.assertTrue(allowed["success"])

    def test_offline_tool_passes_policy_to_command_tool(self):
        seen = {}
        def command(workspace, **kwargs):
            seen.update(kwargs)
            return {"success": False, "return_code": None, "blocked": True,
                    "command": kwargs["command"], "error": "offline blocked"}
        agent = self.make_agent()
        agent.tools["run_command"] = command
        result = agent.execute_tool("run_command", {"command": "pip install package"})
        self.assertTrue(seen["offline"])
        self.assertTrue(result["blocked"])

    def test_online_still_selects_configured_hosted_provider(self):
        agent = MultiModelAgent({"execution_mode": "online", "provider": "ollama",
            "architect_provider": "groq", "models": {"planner": "planner"}}, {}, [], tempfile.gettempdir())
        with patch.dict("os.environ", {"GROQ_API_KEY": "test", "NVIDIA_API_KEY": ""}):
            self.assertEqual(agent._hosted_candidates("planner", "groq"), ["groq"])

    def test_online_connection_failure_does_not_fall_back_in_offline_mode(self):
        agent = self.make_agent()
        agent._architect_task = lambda task, previous_report=None: (
            fallback_specification(task), {"provider": "none", "status": "NOT RUN"})
        calls = []
        def failed_local(task, reports=None, coder_provider=None):
            calls.append(coder_provider)
            agent.last_task_report = {"task": task, "verification_status": "PARTIAL",
                                      "changed_files": [], "commands": [], "static_checks": []}
            raise ConnectionError("Ollama connection refused")
        agent.integrate = failed_local
        with patch("agent.loop.GroqClient", side_effect=AssertionError("Groq called")), \
             patch("agent.loop.NvidiaClient", side_effect=AssertionError("NVIDIA called")):
            result = agent.run("Build a calculator")
        self.assertEqual(calls, [None])
        self.assertIn("selected coding provider failed", result)
        self.assertEqual(agent.last_task_report["verification_status"], "FAIL")

    def test_hinglish_calculator_fallback_has_real_deliverables_without_noise_words(self):
        spec = fallback_specification("calculator folder banao aur usme advanced calculator banao")
        paths = [item["path"] for item in spec["expected_files"]]
        self.assertTrue(all(path.startswith("calculator/") for path in paths))
        self.assertFalse(any("ayr" in path or path.endswith("/ka") for path in paths))
        self.assertEqual(spec["task_type"], "calculator_development")

    def test_generic_new_project_fallback_infers_web_and_api_deliverables(self):
        website = fallback_specification("website banao")
        api = fallback_specification("Build a REST API")
        self.assertEqual(website["task_type"], "website_development")
        self.assertIn("index.html", [item["path"] for item in website["expected_files"]])
        self.assertEqual(api["task_type"], "api_development")
        self.assertEqual([item["path"] for item in api["expected_files"]], ["main.py"])
        self.assertEqual(api["test_commands"], [])

    def test_missing_local_model_error_is_actionable_without_auto_pull(self):
        client = OllamaClient(model="missing-model")
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("404 model not found")))))
        with self.assertRaisesRegex(RuntimeError, "not installed") as caught:
            client.chat([{"role": "user", "content": "hi"}], tools=[{"function": {"name": "test"}}])
        self.assertNotIn("ollama pull", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
