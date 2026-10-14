import sys
from types import SimpleNamespace
from unittest.mock import patch

from agent.loop import MultiModelAgent
from agent.verification import verify_changed_file
from main import offer_provider_recovery
from tools.registry import get_tool_schemas, get_tools


class TimeoutClient:
    model = "deepseek-ai/deepseek-v4.1-flash"

    def chat(self, messages, tools=None):
        raise TimeoutError("simulated NVIDIA request timed out")


def test_nvidia_timeout_creates_and_verifies_car_game_with_real_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    config = {
        "provider": "nvidia",
        "execution_mode": "online",
        "hosted_api_enabled": True,
        "local_only": False,
        "nvidia_model": "deepseek-ai/deepseek-v4.1-flash",
        "architect_provider": "nvidia",
        "task_architect_enabled": True,
        "max_escalations": 0,
        "max_agent_steps": 4,
        "persistent_memory_enabled": False,
        "training_data_collection_enabled": False,
    }
    agent = MultiModelAgent(config, get_tools(), get_tool_schemas(), str(tmp_path))
    agent.get_client = lambda *args, **kwargs: TimeoutClient()

    result = agent.run("Create a playable Python car racing game with GUI.")
    report = agent.last_task_report

    expected = ["main.py", "game_logic.py", "README.md", "requirements.txt", "tests/test_game_logic.py"]
    assert all((tmp_path / path).is_file() for path in expected)
    assert "def move_player" in (tmp_path / "game_logic.py").read_text(encoding="utf-8")
    assert "mainloop" in (tmp_path / "main.py").read_text(encoding="utf-8")
    assert "collision" in (tmp_path / "tests/test_game_logic.py").read_text(encoding="utf-8").lower()
    assert report["verification_status"] == "PASS"
    assert report["recovery_status"] == "PASS"
    assert report["implementation_source"].startswith("deterministic local car-game recovery")
    assert len(report["llm_requests"]) == 2
    assert sum(not item["success"] for item in report["llm_requests"]) == 2
    assert len(report["llm_attempts"]) == 2
    assert len(report["tool_durations"]) == 7
    assert all(item["success"] for item in report["tool_durations"])
    assert report["test_results"]["discovered"] == 4
    assert report["test_results"]["failed"] == 0
    assert report["coder"]["provider"] == "nvidia"
    assert "Verification status: PASS" in result
    assert "LLM calls 2; provider request attempts 2 (0 passed, 2 failed)" in result
    assert "tool operations 7 (7 passed, 0 failed)" in result
    assert agent.workspace == str(tmp_path)


def test_car_game_recovery_is_not_used_for_unrelated_tasks():
    from agent.game_recovery import supports_car_game_recovery

    assert supports_car_game_recovery("create a python in car game with gui")
    assert not supports_car_game_recovery("create a Python calculator GUI")


def test_failed_provider_without_safe_recovery_is_reported_blocked_not_success(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    config = {
        "provider": "nvidia", "execution_mode": "online", "nvidia_model": "test-model",
        "architect_provider": "nvidia", "max_escalations": 0,
        "persistent_memory_enabled": False, "training_data_collection_enabled": False,
    }
    agent = MultiModelAgent(config, get_tools(), get_tool_schemas(), str(tmp_path))
    agent.get_client = lambda *args, **kwargs: TimeoutClient()
    result = agent.run("Create a Python calculator with tests.")
    assert agent.last_task_report["verification_status"] == "FAIL"
    assert len(agent.last_task_report["llm_requests"]) == 2
    assert agent.last_task_report["tool_durations"] == []
    assert not list(tmp_path.iterdir())
    assert "No safe local recovery" in result


def test_provider_recovery_requires_an_explicit_user_choice():
    agent = SimpleNamespace(
        last_task_report={"verification_status": "FAIL", "llm_requests": [{"provider": "nvidia", "success": False}]},
        run=lambda task: "rerun",
    )
    with patch("builtins.input", return_value=""):
        assert offer_provider_recovery(agent, {"provider": "nvidia"}, "task") is None
    with patch("builtins.input", return_value="1"):
        assert offer_provider_recovery(agent, {"provider": "nvidia"}, "task") == "rerun"


def test_local_recovery_choice_explicitly_forces_offline_provider():
    config = {"provider": "nvidia", "execution_mode": "online"}
    agent = SimpleNamespace(
        last_task_report={"verification_status": "FAIL", "llm_requests": [{"provider": "nvidia", "success": False}]},
        run=lambda task: (config["provider"], config["execution_mode"]),
    )
    with patch("main.discover_ollama_models", return_value=["qwen2.5-coder:3b"]), \
         patch("builtins.input", side_effect=["3", "1"]):
        assert offer_provider_recovery(agent, config, "task") == ("ollama", "offline")


def test_alternate_provider_is_used_only_after_explicit_selection():
    config = {"provider": "nvidia", "execution_mode": "online"}
    agent = SimpleNamespace(
        last_task_report={"verification_status": "FAIL", "llm_requests": [{"provider": "nvidia", "success": False}]},
        run=lambda task: config["provider"],
    )
    def select_groq(config):
        config["provider"] = "groq"
        return "groq"

    with patch("builtins.input", return_value="2"), patch("main.choose_provider", side_effect=select_groq):
        assert offer_provider_recovery(agent, config, "task") == "groq"


def test_retry_attempts_are_counted_individually():
    agent = MultiModelAgent({"provider": "nvidia"}, {}, [], ".")
    agent.last_task_report = {"llm_requests": [], "llm_attempts": []}

    class RetryClient:
        model = "nvidia-test-model"
        def chat(self, messages, tools=None):
            agent._record_provider_retry("nvidia", 1, 1, 0.1, TimeoutError("temporary timeout"))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    agent._chat_with_events(RetryClient(), [{"role": "user", "content": "test"}], provider="nvidia")
    assert [attempt["success"] for attempt in agent.last_task_report["llm_attempts"]] == [False, True]
    assert len(agent.last_task_report["llm_requests"]) == 1
    assert agent.last_task_report["llm_requests"][0]["success"] is True


def test_timeout_after_verified_file_changes_preserves_real_execution_evidence(tmp_path):
    agent = MultiModelAgent({
        "execution_mode": "offline", "provider": "ollama", "local_model": "mock",
        "max_escalations": 0, "persistent_memory_enabled": False,
    }, {}, [], str(tmp_path))
    specification = {
        "task_type": "software_change", "expected_files": [
            {"path": "main.py", "responsibility": "Requested source"}
        ], "test_commands": [], "acceptance_criteria": [],
    }
    agent._architect_task = lambda task, previous_report=None: (
        specification, {"provider": "none", "model": "none", "status": "SKIPPED"}
    )

    def timed_out_after_check(task, reports=None, coder_provider=None):
        source = tmp_path / "main.py"
        source.write_text("def answer():\n    return 42\n", encoding="utf-8")
        command = f'"{sys.executable}" -m py_compile main.py'
        agent.last_task_report = {
            "task": task, "changed_files": ["main.py"], "static_checks": [
                verify_changed_file(str(tmp_path), "main.py")
            ], "commands": [{
                "command": command, "return_code": 0, "success": True,
                "meaningful_check": True, "stdout": "", "stderr": "",
            }], "verification_status": "PARTIAL",
        }
        raise TimeoutError("local model stalled after verification")

    agent.integrate = timed_out_after_check
    result = agent.run("Fix main.py and verify it.")
    assert agent.last_task_report["changed_files"] == ["main.py"]
    assert agent.last_task_report["commands"][0]["return_code"] == 0
    assert agent.last_task_report["verification_status"] == "PASS"
    assert "TimeoutError" in result



def test_reported_provider_failure_triggers_recovery_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    config = {
        "provider": "nvidia",
        "execution_mode": "online",
        "hosted_api_enabled": True,
        "local_only": False,
        "nvidia_model": "deepseek-ai/deepseek-v4.1-flash",
        "architect_provider": "nvidia",
        "task_architect_enabled": True,
        "max_escalations": 0,
        "persistent_memory_enabled": False,
        "training_data_collection_enabled": False,
    }
    agent = MultiModelAgent(config, get_tools(), get_tool_schemas(), str(tmp_path))
    specification = {
        "expected_files": [{"path": "main.py", "responsibility": "Playable car game"}],
        "test_commands": ["python -m unittest discover -s tests -v"],
        "acceptance_criteria": ["Playable game with movement and collisions"],
    }
    agent._architect_task = lambda task, previous_report=None: (
        specification, {"provider": "nvidia", "model": "deepseek-ai/deepseek-v4.1-flash", "status": "FAILED"}
    )

    def return_failed_report(task, orchestration, coder_provider=None):
        agent.last_task_report = {
            "task": task,
            "specification": specification,
            "llm_requests": [{
                "provider": "nvidia",
                "model": "deepseek-ai/deepseek-v4.1-flash",
                "success": False,
                "error": "Request timed out",
            }],
            "changed_files": [],
            "commands": [],
            "static_checks": [],
            "tool_durations": [],
            "verification_status": "FAIL",
            "initial_error": "Request timed out",
        }
        return "Task finished: Not verified"

    agent.integrate = return_failed_report
    result = agent.run("Create a playable Python car racing game with GUI.")

    expected = ["main.py", "game_logic.py", "README.md", "requirements.txt", "tests/test_game_logic.py"]
    assert all((tmp_path / path).is_file() for path in expected)
    assert agent.last_task_report["verification_status"] == "PASS"
    assert agent.last_task_report["recovery_status"] == "PASS"
    assert "Verification status: PASS" in result
