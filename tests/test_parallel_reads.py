import json
import threading
import time
from types import SimpleNamespace

from agent.dashboard import EventBus
from agent.loop import MultiModelAgent


def _call(name, arguments, number):
    return SimpleNamespace(id=f"call-{number}", function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def test_independent_read_tools_run_concurrently_and_preserve_order(tmp_path):
    active = 0
    peak = 0
    lock = threading.Lock()
    rendezvous = threading.Barrier(2)

    def read_file(workspace, path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        rendezvous.wait(timeout=2)
        time.sleep(0.02)
        with lock:
            active -= 1
        return f"contents:{path}"

    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    agent = MultiModelAgent({"provider": "ollama", "max_parallel_tool_workers": 2},
                            {"read_file": read_file}, [], str(tmp_path), event_bus=bus)
    calls = [_call("read_file", {"path": "a.py"}, 1), _call("read_file", {"path": "b.py"}, 2)]
    result = agent._run_independent_reads(calls)
    assert peak == 2
    assert [result[i]["tool_call_id"] for i in range(2)] == ["call-1", "call-2"]
    assert [item["event"] for item in events].count("worker.started") == 2
    assert [item["event"] for item in events].count("worker.finished") == 2
    assert agent.last_task_report["performance"][0]["workers"] == 2


def test_mutations_or_single_llm_calls_use_sequential_path(tmp_path):
    agent = MultiModelAgent({"provider": "ollama", "max_parallel_tool_workers": 4}, {}, [], str(tmp_path))
    calls = [_call("write_file", {"path": "a", "content": "x"}, 1), _call("read_file", {"path": "b"}, 2)]
    assert agent._run_independent_reads(calls) == {}
    assert agent._run_independent_reads(calls[:1]) == {}


def test_failed_worker_reports_failure_without_losing_tool_result(tmp_path):
    def read_file(workspace, path):
        raise OSError("cannot read file")

    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    agent = MultiModelAgent({"provider": "ollama", "max_parallel_tool_workers": 2},
                            {"read_file": read_file}, [], str(tmp_path), event_bus=bus)
    calls = [_call("read_file", {"path": "a.py"}, 1), _call("read_file", {"path": "b.py"}, 2)]
    result = agent._run_independent_reads(calls)
    assert all(json.loads(result[i]["content"])["success"] is False for i in range(2))
    finished = [event for event in events if event["event"] == "worker.finished"]
    assert len(finished) == 2 and all(event["status"] == "FAILED" for event in finished)


def test_agent_loop_limit_is_configurable_and_bounded(tmp_path):
    assert MultiModelAgent({"max_agent_steps": 9}, {}, [], str(tmp_path)).max_steps == 9
    assert MultiModelAgent({"max_agent_steps": 1000}, {}, [], str(tmp_path)).max_steps == 80


def test_total_worker_budget_caps_tool_and_llm_concurrency(tmp_path):
    agent = MultiModelAgent({
        "max_total_workers": 3,
        "max_parallel_tool_workers": 4,
        "max_concurrent_llm_requests": 2,
    }, {}, [], str(tmp_path))
    assert agent.max_workers == 2
    assert agent.max_concurrent_llm_requests == 1
    assert agent.max_workers + agent.max_concurrent_llm_requests <= agent.max_total_workers


def test_task_deadline_stops_before_another_model_request(tmp_path):
    agent = MultiModelAgent({"provider": "ollama", "task_deadline_seconds": 30}, {}, [], str(tmp_path))
    agent.get_client = lambda *args, **kwargs: SimpleNamespace(model="mock")
    agent._active_task_deadline = time.monotonic() - 1
    result = agent.integrate("Create a small app", {
        "specification": {"expected_files": [], "acceptance_criteria": [], "test_commands": []}
    })
    assert "Task deadline reached after 30 seconds" in result
    assert agent.last_task_report["verification_status"] == "BLOCKED"


def test_read_worker_timeout_returns_failure_and_requests_cancellation(tmp_path):
    def slow_read(workspace, path):
        time.sleep(0.2)
        return "finished"

    bus = EventBus()
    events = []
    bus.subscribe(events.append)
    agent = MultiModelAgent({"provider": "ollama", "max_parallel_tool_workers": 2,
                             "parallel_tool_timeout_seconds": 0.1},
                            {"read_file": slow_read}, [], str(tmp_path), event_bus=bus)
    calls = [_call("read_file", {"path": "a.py"}, 1), _call("read_file", {"path": "b.py"}, 2)]
    result = agent._run_independent_reads(calls)
    assert all("timed out" in json.loads(result[i]["content"])["error"] for i in range(2))
    assert all(event["status"] == "FAILED" for event in events if event["event"] == "worker.finished")


def test_llm_concurrency_semaphore_enforces_its_separate_limit(tmp_path):
    active = 0
    peak = 0
    lock = threading.Lock()
    class Client:
        model = "mock"
        def chat(self, messages, tools=None):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return "ok"
    agent = MultiModelAgent({"provider": "ollama", "max_concurrent_llm_requests": 1}, {}, [], str(tmp_path))
    threads = [threading.Thread(target=agent._chat_with_events, args=(Client(), [], None, "ollama")) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in threads)
    assert peak == 1
    assert len(agent.last_task_report["llm_requests"]) == 2
