import json
from unittest.mock import patch

from agent.context import WorkspaceContextManager
from agent.loop import MultiModelAgent
from agent.workspace_memory import MEMORY_FILES, WorkspaceMemory
from training.trajectory import build_trajectory_record


def make_memory(root, **config):
    context = WorkspaceContextManager(root, max_context_chars=6000)
    memory = WorkspaceMemory(root, context, {"persistent_memory_enabled": True, **config})
    return context, memory


def test_initial_discovery_summary_index_and_secret_exclusion(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("# Notes App\nRun with python src/main.py\n", encoding="utf-8")
    (tmp_path / "src" / "main.py").write_text('"""Application entry point."""\nimport json\ndef main(): pass\n', encoding="utf-8")
    (tmp_path / "src" / "notes.py").write_text('"""api_key=hidden-secret-value"""\n', encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=supersecretvalue", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("flask==3.0\n", encoding="utf-8")
    context, memory = make_memory(tmp_path)

    result = memory.refresh()
    paths = [item["path"] for item in memory._data["file_index"]["files"]]
    assert result["indexed_files"] == 4
    assert ".env" not in paths
    memory_files = list((tmp_path / ".localcodex" / "memory").glob("*"))
    assert "supersecretvalue" not in " ".join(path.read_text(encoding="utf-8") for path in memory_files if path.is_file())
    assert "hidden-secret-value" not in " ".join(path.read_text(encoding="utf-8") for path in memory_files if path.suffix == ".json")
    assert "Notes App" in memory._data["project_summary"]
    assert any("Application entry point" in item["responsibility"] for item in memory._data["architecture"]["modules"])
    assert memory._data["architecture"]["modules"][0]["relationship_confidence"] == "inferred"
    assert memory._data["commands"]["items"][0]["verification"] == "unknown"
    assert all((tmp_path / ".localcodex" / "memory" / name).is_file() for name in MEMORY_FILES)
    assert ".localcodex/memory" not in context.build_context("main.py")


def test_memory_persists_across_instances_and_is_workspace_scoped(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir(); second.mkdir()
    (first / "calculator.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    (_, memory1) = make_memory(first)
    memory1.refresh()
    memory1.record_task_start("calculator fails", "task-1")
    memory1.record_command({"command": "pytest", "return_code": 1, "success": False, "stderr": "1 failed", "cwd": str(first)})
    memory1.record_task_finish({"task_id": "task-1", "task": "calculator fails", "verification_status": "PARTIAL",
                                "changed_files": ["calculator.py"], "commands": [{"command": "pytest", "return_code": 1, "success": False, "stderr": "1 failed"}],
                                "completeness_issues": ["test failed"]})
    _, memory2 = make_memory(first)
    memory2.refresh()
    assert memory2._data["unresolved_issues"]["items"][-1]["task_id"] == "task-1"
    assert "1 failed" in json.dumps(memory2._data["commands"])
    assert "task-1" in memory2.context_for("pichla code is still failing")
    assert "calculator" not in memory2.context_for("website CSS colors").casefold()
    memory2.record_task_start("resolve calculator failure", "task-2")
    memory2.record_task_finish({"task_id": "task-2", "task": "resolve calculator failure",
                                "verification_status": "PASS", "changed_files": ["calculator.py"],
                                "commands": [], "static_checks": []})
    assert memory2._data["unresolved_issues"]["items"][-1]["status"] == "resolved"
    _, other = make_memory(second)
    other.refresh()
    assert other.workspace_id != memory2.workspace_id
    assert other.context_for("calculator error") == other.context_for("calculator error")
    assert "calculator.py" not in other.context_for("calculator error")


def test_incremental_refresh_detects_modifications_deletions_and_renames(tmp_path):
    old = tmp_path / "old.py"
    old.write_text("def old(): return 1\n", encoding="utf-8")
    _, memory = make_memory(tmp_path)
    memory.refresh()
    old.unlink()
    (tmp_path / "new.py").write_text("def fresh(): return 2\n", encoding="utf-8")
    result = memory.refresh()
    assert "old.py" in result["deleted"]
    assert "new.py" in result["changed"]
    assert result["stale"] is False
    assert result["stale_detected"] is True
    assert memory._data["manifest"]["project_fingerprint"]


def test_corrupt_json_is_rebuilt_and_context_budget_is_bounded(tmp_path):
    (tmp_path / "feature.py").write_text("def calculate_total(items): return sum(items)\n", encoding="utf-8")
    _, first = make_memory(tmp_path)
    first.refresh()
    index_path = tmp_path / ".localcodex" / "memory" / "file_index.json"
    index_path.write_text("{broken", encoding="utf-8")
    _, recovered = make_memory(tmp_path)
    assert recovered.stale
    recovered.refresh()
    assert json.loads(index_path.read_text(encoding="utf-8"))["files"]
    context = recovered.context_for("calculate total", max_chars=1200)
    assert len(context) <= 1200
    assert "feature.py" in context


def test_memory_directory_path_cannot_escape_workspace(tmp_path):
    _, memory = make_memory(tmp_path, workspace_memory_dir="../outside-memory")
    assert not memory.enabled
    assert not (tmp_path.parent / "outside-memory").exists()


def test_disabled_memory_creates_nothing_and_storage_failures_do_not_crash(tmp_path):
    _, disabled = make_memory(tmp_path, persistent_memory_enabled=False)
    assert not disabled.enabled
    assert not (tmp_path / ".localcodex").exists()
    _, memory = make_memory(tmp_path)
    with patch("agent.workspace_memory._atomic_write", side_effect=OSError("read only")):
        memory.record_task_start("safe failure", "task-x")
    assert memory.status.startswith("Memory write failed")


def test_offline_memory_indexing_and_retrieval_never_construct_hosted_clients(tmp_path):
    (tmp_path / "app.py").write_text("def run(): return 'local'\n", encoding="utf-8")
    agent = MultiModelAgent({"execution_mode": "offline", "persistent_memory_enabled": True}, {}, [], str(tmp_path))
    with patch("agent.loop.GroqClient", side_effect=AssertionError("Groq must not be used")), \
         patch("agent.loop.NvidiaClient", side_effect=AssertionError("NVIDIA must not be used")):
        agent.workspace_memory.refresh()
        retrieved = agent.workspace_memory.context_for("run app")
    assert "app.py" in retrieved
    assert agent.workspace_memory.enabled


def test_trajectory_contains_compact_context_and_labels_unverified_results():
    record = build_trajectory_record({"task": "Fix calculator", "verification_status": "PARTIAL",
        "context_snapshot": "src/calculator.py: inspected", "memory_context": {"retrieved_paths": ["src/calculator.py"]},
        "commands": [{"command": "pytest", "return_code": 1, "stderr": "token=secret-value"}]}, "workspace-id")
    assert record["retrieved_paths"] == ["src/calculator.py"]
    assert record["failure_category"] == "incomplete_or_unverified"
    assert "secret-value" not in json.dumps(record)


def test_agent_loads_relevant_context_and_persists_verified_task_trajectory(tmp_path):
    (tmp_path / "calculator.py").write_text("def add(left, right): return left + right\n", encoding="utf-8")
    agent = MultiModelAgent({"execution_mode": "offline", "provider": "ollama", "persistent_memory_enabled": True}, {}, [], str(tmp_path))
    agent._architect_task = lambda task, previous_report=None: ({"acceptance_criteria": ["Add sums"], "expected_files": []}, {"provider": "none", "status": "NOT RUN"})

    def fake_integrate(task, orchestration, coder_provider=None):
        agent.last_task_report = {"task_id": agent.current_task_id, "task": task, "changed_files": ["calculator.py"],
                                  "commands": [], "static_checks": [], "verification_status": "PASS"}
        context = agent._workspace_context_text(task, specification=orchestration["specification"])
        assert "def add(left, right)" in context
        assert "calculator.py" in agent.last_task_report["memory_context"]["retrieved_paths"]
        return "verified"

    agent.integrate = fake_integrate
    result = agent.run("add calculator behavior")
    assert result == "verified", agent.last_task_report.get("initial_error")
    memory_dir = tmp_path / ".localcodex" / "memory"
    tasks = json.loads((memory_dir / "active_tasks.json").read_text(encoding="utf-8"))["items"]
    assert tasks[-1]["verification_state"] == "PASS"
    trajectory = json.loads((memory_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert trajectory["verification_status"] == "PASS"
    assert trajectory["workspace_id"] == agent.workspace_memory.workspace_id
    assert "calculator.py" in trajectory["retrieved_paths"]
    assert "def add(left, right)" in trajectory["source_changes"]["calculator.py"]
