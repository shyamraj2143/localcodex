import io
import pytest

from agent.dashboard import TerminalDashboard
from agent.loop import MultiModelAgent
from tools.files import list_files, write_file


def test_dashboard_reports_actual_file_edits_and_verification(tmp_path):
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace=str(tmp_path), stream=output)
    agent = MultiModelAgent(
        config={"provider": "ollama"},
        tools={
            "read_file": lambda workspace, path: (tmp_path / path).read_text(encoding="utf-8"),
            "write_file": lambda workspace, path, content: _write(tmp_path / path, content),
            "edit_file": lambda workspace, path, old_text, new_text: _edit(tmp_path / path, old_text, new_text),
        },
        tool_schemas=[], workspace=str(tmp_path), event_bus=dashboard.bus,
    )
    dashboard.bus.emit("task.started", task="write and edit", workspace=str(tmp_path))
    assert agent.execute_tool("write_file", {"path": r"nested\game.py", "content": "one\ntwo\n"})["success"]
    assert agent.execute_tool("edit_file", {"path": r"nested\game.py", "old_text": "two", "new_text": "two\nthree"})["success"]
    content = agent.execute_tool("read_file", {"path": r"nested\game.py"})
    assert content == "one\ntwo\nthree\n"
    dashboard.bus.emit("verification", message="pytest — PASS")
    dashboard.bus.emit("workspace.memory", status="Refreshed", indexed_files=11,
                       context_files=3, unresolved_issues=1, last_task="Calculator GUI repair", stale=False,
                       project_type="Python application")
    dashboard.bus.emit("task.finished", status="PASS")

    file = dashboard.files["nested/game.py"]
    assert file == {"action": "read", "actions": ["created", "modified", "read"],
                    "lines": 3, "added": 3, "removed": 0, "summary": "Read"}
    assert "nested/game.py" in output.getvalue()
    assert "pytest — PASS" in output.getvalue()
    assert dashboard.memory["indexed_files"] == 11
    assert dashboard.memory["last_task"] == "Calculator GUI repair"


def test_dashboard_tracks_files_deleted_by_a_real_command(tmp_path):
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace=str(tmp_path), stream=output)
    write_file(str(tmp_path), "game.py", "print('game')\n")
    agent = MultiModelAgent(
        config={"provider": "ollama"},
        tools={
            "list_files": list_files,
            "run_command": lambda workspace, command: (
                (tmp_path / "game.py").unlink() or {"success": True, "return_code": 0}
            ),
        },
        tool_schemas=[], workspace=str(tmp_path), event_bus=dashboard.bus,
    )
    assert agent.execute_tool("run_command", {"command": "remove game.py"})["success"]
    assert dashboard.files["game.py"]["action"] == "deleted"
    assert dashboard.files["game.py"]["removed"] == 1


def test_rich_dashboard_starts_and_finishes_without_a_tty():
    class TerminalBuffer(io.StringIO):
        def isatty(self):
            return True

    output = TerminalBuffer()
    dashboard = TerminalDashboard(mode="rich", workspace="workspace", provider="mock", model="test", stream=output)
    dashboard.bus.emit("task.started", task="smoke", workspace="workspace")
    dashboard.bus.emit("model.request", provider="mock", model="test")
    dashboard.bus.emit("model.response", duration=0.01, success=True)
    dashboard.bus.emit("task.finished", status="PASS")
    assert "smoke" in output.getvalue()
    assert "Finished" in output.getvalue()


def test_dashboard_does_not_display_internal_step_counters():
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", stream=output)
    dashboard.bus.emit("task.started", task="smoke")
    dashboard.bus.emit("step.started", step=4, limit=24, phase="model")
    dashboard.bus.emit("activity", message="Running tests")
    dashboard.bus.emit("task.finished", status="PASS")
    assert "Step 4 of 24" not in output.getvalue()
    assert "step 4/24" not in output.getvalue()


def test_dashboard_shows_task_phases_and_model_wait_in_plain_mode():
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", stream=output)
    dashboard.bus.emit("task.started", task="build a game")
    dashboard.bus.emit("activity", message="Inspecting workspace and loading project context")
    dashboard.bus.emit("activity", message="Planning task and identifying deliverables")
    dashboard.bus.emit("model.request", provider="ollama", phase="planning")
    dashboard.bus.emit("tool.started", tool="list_files", label="Listing workspace files")

    display = output.getvalue()
    assert "Inspecting workspace" in display
    assert "Planning task" in display
    assert "planning the task" in display
    assert "Looking through the project files" in display
    assert dashboard.activity_log[-1]["message"] == "Looking through the project files."


def test_dashboard_surfaces_real_model_request_error():
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", stream=output)
    dashboard.bus.emit("model.response", success=False, duration=0.2,
                       error="NVIDIA request timed out.")
    assert "NVIDIA request timed out." in dashboard.current_reason
    assert any("NVIDIA request timed out." in entry for entry in dashboard.technical_log)


def test_dashboard_shows_provider_retry_reason():
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", stream=io.StringIO())
    dashboard.bus.emit("provider.retry", provider="NVIDIA", attempt=1, attempts=1,
                       delay=1.5, error="The read operation timed out")
    assert "retrying" in dashboard.activity.lower()
    assert "The read operation timed out" in dashboard.current_reason


def test_dashboard_shows_worker_failure_and_reason_without_inventing_activity():
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", stream=output)
    dashboard.bus.emit("task.started", task="inspect files")
    dashboard.bus.emit("worker.queued", worker_id="read-1", task="src/main.py",
                       operation="read_file", status="QUEUED", dependencies=["read-0"])
    dashboard.bus.emit("worker.started", worker_id="read-1", task="src/main.py",
                       operation="read_file", status="RUNNING")
    dashboard.bus.emit("worker.finished", worker_id="read-1", status="FAILED",
                       duration=0.25, summary="Permission denied")

    worker = dashboard.workers["read-1"]
    assert worker["status"] == "FAILED"
    assert worker["duration"] == 0.25
    assert worker["summary"] == "Permission denied"
    assert worker["dependencies"] == ["read-0"]
    assert "active worker(s)" not in output.getvalue()


def test_dashboard_labels_unrun_checks_and_failed_reads_honestly():
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", stream=output)
    dashboard.bus.emit("task.started", task="inspect a missing file")
    dashboard.bus.emit("tool.started", tool="read_file", arguments={"path": "missing.py"})
    dashboard.bus.emit("tool.finished", tool="read_file", duration=0.01, success=False)
    dashboard.bus.emit("tool.result", tool="read_file", result={"success": False, "error": "file not found"})
    dashboard.bus.emit("task.report", report={"verification_status": "NOT VERIFIED", "commands": [],
                                                "completeness_issues": ["Missing required file: missing.py"]})
    dashboard.bus.emit("task.finished", status="FAIL")

    rendered = output.getvalue()
    assert "The file could not be read" in rendered
    assert "file not found" in rendered
    assert "Not verified" in rendered
    assert "Not run or not reported" in rendered


def test_rich_demo_dashboard_renders_workflow_and_hides_technical_details_by_default():
    pytest.importorskip("rich")
    from rich.console import Console

    dashboard = TerminalDashboard(mode="plain", workspace="workspace", demo_mode=True,
                                  show_technical=False, stream=io.StringIO())
    dashboard.bus.emit("task.started", task="inspect and update a file")
    dashboard.rich_enabled = True
    dashboard.bus.emit("activity", message="Inspecting workspace and loading project context")
    dashboard.bus.emit("tool.started", tool="list_files")
    rendered = io.StringIO()
    Console(file=rendered, width=100).print(dashboard._render())

    text = rendered.getvalue()
    assert "Current task" in text
    assert "Workflow" in text
    assert "Inspect workspace" in text
    assert "Looking through the project files" in text
    assert "list_files" not in text
    assert "Workspace memory" not in text


def test_hinglish_presentation_and_optional_technical_log():
    output = io.StringIO()
    dashboard = TerminalDashboard(mode="plain", workspace="workspace", language="hinglish",
                                  show_technical=True, stream=output)
    dashboard.bus.emit("task.started", task="inspect files")
    dashboard.bus.emit("tool.started", tool="list_files")
    dashboard.bus.emit("tool.finished", tool="list_files", duration=0.12, success=True)
    dashboard.bus.emit("task.finished", status="NOT VERIFIED")

    assert "Project ki files inspect kar raha hai" in output.getvalue()
    assert "tool.started" in output.getvalue()
    assert dashboard.verification_status == "Not verified"


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"success": True, "path": str(path)}


def _edit(path, old_text, new_text):
    current = path.read_text(encoding="utf-8")
    if old_text not in current:
        return {"success": False, "error": "not found"}
    path.write_text(current.replace(old_text, new_text, 1), encoding="utf-8")
    return {"success": True, "path": str(path)}
