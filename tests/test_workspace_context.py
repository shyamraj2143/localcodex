from pathlib import Path

from agent.context import WorkspaceContextManager


def test_context_respects_gitignore_and_includes_project_metadata(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored/\n*.secret\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Run the app with python main.py", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pygame-ce>=2.5.5", encoding="utf-8")
    (tmp_path / "main.py").write_text("def main():\n    launch_gui()\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "secret.py").write_text("private", encoding="utf-8")
    (tmp_path / "dump.secret").write_text("private", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=should-not-be-indexed", encoding="utf-8")

    context = WorkspaceContextManager(tmp_path).build_context("GUI does not open")

    assert "requirements.txt" in context
    assert "pygame-ce: MISSING from active Python environment" in context
    assert "main.py" in context
    assert "launch_gui" in context
    assert "ignored/secret.py" not in context
    assert "should-not-be-indexed" not in context


def test_context_refreshes_changed_files_and_limits_prompt_size(tmp_path):
    source = tmp_path / "gui.py"
    source.write_text("def launch():\n    return 'old'\n", encoding="utf-8")
    manager = WorkspaceContextManager(tmp_path, max_context_chars=5000)
    first = manager.build_context("GUI launch fails")
    source.write_text("def launch():\n    return 'fixed'\n", encoding="utf-8")
    second = manager.build_context("GUI launch fails")

    assert "old" in first
    assert "fixed" in second
    assert len(second) <= 5000


def test_context_records_python_structure_git_and_prior_execution(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "game.py").write_text(
        "import pygame\nclass Game:\n    def start(self):\n        pass\n", encoding="utf-8"
    )
    report = {
        "task": "Build the game", "changed_files": ["src/game.py"],
        "verification_status": "PARTIAL", "commands": [{"command": "pytest", "stderr": "1 failed"}],
        "specification": {"expected_files": [{"path": "src/missing_asset.png"}]},
    }
    context = WorkspaceContextManager(tmp_path).build_context(
        "this game test failed", previous_report=report,
        recent_tool_calls=[{"tool": "run_command", "result": {"success": False}}],
    )
    assert "WORKSPACE ROOT:" in context
    assert "Imports: pygame" in context
    assert "ClassDef Game" in context
    assert "verification_status" in context
    assert "run_command" in context
    assert "src/missing_asset.png" in context
    assert "MISSING REQUIRED ARTIFACTS" in context
