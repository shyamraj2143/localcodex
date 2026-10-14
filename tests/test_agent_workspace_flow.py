from types import SimpleNamespace

from agent.loop import MultiModelAgent
from agent.loop import _test_counts
from agent.orchestration import fallback_specification
from tools.files import read_file, write_file


class CapturingClient:
    model = "mock-coder"

    def __init__(self, content="Answer from inspected code"):
        self.content = content
        self.messages = []
        self.tools_values = []

    def chat(self, messages, tools=None):
        self.messages.append(messages)
        self.tools_values.append(tools)
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_relevant_file_contents_reach_coding_model_without_scanning_all_files(tmp_path):
    write_file(str(tmp_path), "game/gui.py", "def run_gui():\n    return 'window started'\n")
    agent = MultiModelAgent(
        {"provider": "groq", "models": {"coder": "mock"}}, {}, [], str(tmp_path)
    )
    client = CapturingClient()
    agent.get_client = lambda *args, **kwargs: client

    agent.integrate("Why did this GUI window fail?", {"specification": fallback_specification("Fix GUI")})

    prompt = "\n".join(message["content"] for call in client.messages for message in call)
    assert "game/gui.py" in prompt
    assert "window started" in prompt
    assert "WORKSPACE ROOT:" in prompt
    assert "run_command" in prompt


def test_explanation_uses_workspace_evidence_without_calling_tools_or_mutating(tmp_path):
    write_file(str(tmp_path), "app.py", "def calculate():\n    return 42\n")
    before = read_file(str(tmp_path), "app.py")
    agent = MultiModelAgent({"provider": "ollama"}, {}, [], str(tmp_path))
    client = CapturingClient("calculate returns 42")
    agent.get_client = lambda *args, **kwargs: client

    answer = agent.explain("Explain calculate in app.py; do not change code")

    assert answer == "calculate returns 42"
    assert client.tools_values == [None]
    assert "def calculate" in client.messages[0][1]["content"]
    assert read_file(str(tmp_path), "app.py") == before


def test_followup_carries_previous_report_and_tool_evidence(tmp_path):
    agent = MultiModelAgent({"provider": "ollama", "max_escalations": 0}, {}, [], str(tmp_path))
    previous_report = {
        "task": "Build a game", "changed_files": ["game/main.py"],
        "commands": [{"command": "pytest", "stderr": "GUI import failed"}],
        "verification_status": "PARTIAL",
    }
    agent.last_task_report = previous_report
    agent.session_history = [{"role": "user", "content": "Build a game"}]
    agent.recent_tool_calls = [{"tool": "run_command", "result": {"stderr": "GUI import failed"}}]
    seen = {}
    spec = fallback_specification("Fix the game GUI")
    agent._architect_task = lambda task, previous_report=None: (
        spec, {"provider": "none", "model": "none", "status": "SKIPPED"}
    )

    def integrate(task, reports=None, coder_provider=None):
        seen.update(reports)
        agent.last_task_report = {"task": task, "verification_status": "PASS", "changed_files": [], "commands": [], "static_checks": []}
        return "investigated the existing game"

    agent.integrate = integrate
    result = agent.run("is game me main.py run karne par gui open nahi hua")

    assert "investigated" in result
    assert seen["previous_task_report"]["changed_files"] == ["game/main.py"]
    assert "GUI import failed" in str(seen["previous_task_report"])
    assert agent.recent_tool_calls


def test_short_followup_bug_keeps_prior_report_even_without_a_file_name(tmp_path):
    agent = MultiModelAgent({"provider": "ollama", "max_escalations": 0}, {}, [], str(tmp_path))
    agent.last_task_report = {"task": "Build a GUI", "changed_files": ["gui.py"], "verification_status": "PARTIAL"}
    agent.session_history = [{"role": "assistant", "content": "GUI created"}]
    spec = fallback_specification("Fix the GUI")
    agent._architect_task = lambda task, previous_report=None: (
        spec, {"provider": "none", "model": "none", "status": "SKIPPED"}
    )
    seen = {}
    def integrate(task, reports=None, coder_provider=None):
        seen.update(reports)
        agent.last_task_report = {"task": task, "verification_status": "NOT VERIFIED", "changed_files": [], "commands": [], "static_checks": []}
        return "checked existing GUI"
    agent.integrate = integrate

    agent.run("button kaam nahi kar raha")

    assert seen["previous_task_report"]["changed_files"] == ["gui.py"]


def test_unrelated_new_task_drops_old_conversation_context(tmp_path):
    agent = MultiModelAgent({"provider": "ollama", "max_escalations": 0}, {}, [], str(tmp_path))
    agent.last_task_report = {"task": "Build a game", "changed_files": ["game/main.py"]}
    agent.session_history = [{"role": "user", "content": "Build a game"}]
    agent.recent_tool_calls = [{"tool": "write_file", "arguments": {"path": "game/main.py"}}]
    spec = fallback_specification("Build a calculator app")
    agent._architect_task = lambda task, previous_report=None: (
        spec, {"provider": "none", "model": "none", "status": "SKIPPED"}
    )
    seen = {}

    def integrate(task, reports=None, coder_provider=None):
        seen.update(reports)
        agent.last_task_report = {"task": task, "verification_status": "NOT VERIFIED", "changed_files": [], "commands": [], "static_checks": []}
        return "started unrelated task"

    agent.integrate = integrate
    agent.run("Build a calculator app")

    assert seen["previous_task_report"] == {}
    assert agent.session_history == []
    assert agent.recent_tool_calls == []


def test_command_stderr_is_retained_for_the_next_debugging_turn(tmp_path):
    agent = MultiModelAgent(
        {"provider": "ollama"},
        {"run_command": lambda workspace, command: {
            "success": False, "command": command, "cwd": workspace,
            "return_code": 2, "stdout": "", "stderr": "ModuleNotFoundError: pygame",
            "duration_seconds": 0.04, "timed_out": False,
        }},
        [], str(tmp_path),
    )
    agent.execute_tool("run_command", {"command": "python game/main.py"})
    context = agent._workspace_context_text("GUI does not open")
    assert "ModuleNotFoundError: pygame" in context
    assert "python game/main.py" in context


def test_test_counts_require_actual_runner_output():
    assert _test_counts("python -m pytest", "4 passed, 1 failed", "") == {
        "discovered": 5, "passed": 4, "failed": 1,
    }
    assert _test_counts("python -m unittest", "Ran 0 tests in 0.0s\nOK", "")["discovered"] == 0
    assert _test_counts("python --version", "Python 3.14", "")["discovered"] is None
