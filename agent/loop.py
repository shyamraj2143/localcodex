"""Agentic coding loop for Local Codex.

Unlike the previous planner + parallel-worker pipeline, this loop keeps one
coding agent in control of inspect -> edit -> run -> diagnose -> verify. This
is more reliable for small local Ollama models and mirrors the core workflow
of an interactive coding agent.
"""

import json
import difflib
from concurrent.futures import ThreadPoolExecutor, wait
import threading
from datetime import datetime, timezone
import os
import re
import sys
import time
import uuid
from types import SimpleNamespace

from agent.groq import GroqClient
from agent.context import WorkspaceContextManager
from agent.game_recovery import car_game_files, supports_car_game_recovery
from agent.calculator_recovery import calculator_files, calculator_project_files, supports_calculator_gui, supports_calculator_project
from agent.nvidia import NvidiaClient
from agent.ollama import OllamaClient
from agent.orchestration import completeness_issues, fallback_specification, parse_specification, redact_secrets
from agent.provider_support import safe_error
from agent.request import classify_request, references_existing_context
from agent.verification import verify_changed_file
from agent.workspace_memory import WorkspaceMemory


MAX_TOOL_RESULT_CHARS = 16000
MAX_SESSION_TURNS = 8
MAX_COMMAND_LINE_COUNT_FILE_BYTES = 512_000
MAX_COMMAND_LINE_COUNT_SCAN_BYTES = 8_000_000


def _dashboard_line_delta(before, after):
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=before.splitlines(), b=after.splitlines(), autojunk=False
    ).get_opcodes():
        if tag in {"insert", "replace"}:
            added += j2 - j1
        if tag in {"delete", "replace"}:
            removed += i2 - i1
    return added, removed


def _iter_json_objects(content):
    """Yield JSON objects embedded in plain text, including multiple fenced blocks."""
    if not isinstance(content, str) or not content.strip():
        return

    decoder = json.JSONDecoder()
    seen = set()
    for match in re.finditer(r"\{", content):
        start = match.start()
        try:
            parsed, end = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        fingerprint = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield parsed


def _extract_json_object(content):
    """Parse the first JSON object from plain text or a fenced code block."""
    return next(_iter_json_objects(content), None)


def recover_text_tool_calls(content, available_tools):
    """Convert JSON-shaped tool calls embedded anywhere in text into executable calls."""
    recovered = []
    seen_calls = set()
    for data in _iter_json_objects(content):
        calls = data.get("tool_calls")
        if not isinstance(calls, list):
            calls = [data]

        for item in calls:
            if not isinstance(item, dict):
                continue

            function_data = item.get("function")
            if not isinstance(function_data, dict):
                function_data = {}

            name = item.get("name") or function_data.get("name")
            arguments = item.get("arguments", function_data.get("arguments", {}))

            if not isinstance(name, str) or name not in available_tools:
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue

            if arguments == []:
                arguments = {}
            if not isinstance(arguments, dict):
                continue

            signature = (
                name,
                json.dumps(arguments, sort_keys=True, ensure_ascii=False),
            )
            if signature in seen_calls:
                continue
            seen_calls.add(signature)
            recovered.append(
                SimpleNamespace(
                    id="call_" + uuid.uuid4().hex,
                    type="function",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
    return recovered

def _is_tool_call_shaped_json(content):
    data = _extract_json_object(content)
    return isinstance(data, dict) and (
        "name" in data or "function" in data or "tool_calls" in data
    )


def _is_tool_result_json(content):
    """Detect a tool result that a small model mistakenly echoes as its answer."""
    data = _extract_json_object(content)
    if not isinstance(data, dict) or not isinstance(data.get("success"), bool):
        return False
    result_fields = {
        "message", "path", "absolute_path", "bytes_written",
        "return_code", "stdout", "stderr", "error", "output",
    }
    return bool(result_fields.intersection(data))


def _is_meaningful_verification_command(command):
    """Reject unrelated successful commands (for example, pip --version) as code verification."""
    if not isinstance(command, str):
        return False
    command = command.lower()
    patterns = (
        r"\bpytest\b", r"\bunittest\b", r"\bcompileall\b",
        r"\bpy_compile\b", r"\bnode\s+--check\b",
        r"\bnpm\s+(?:test|run\s+(?:test|build|lint|typecheck))\b",
        r"\bnpx\s+(?:tsc|eslint|vitest|jest)\b",
        r"\btsc\s+--noemit\b", r"\beslint\b", r"\bprettier\s+--check\b",
        r"\bruff\s+check\b", r"\bmypy\b", r"\bgo\s+test\b",
        r"\bcargo\s+test\b", r"\bmvn\s+test\b", r"\bgradle\s+test\b",
        r"\bphp\s+-l\b", r"\bdotnet\s+test\b",
    )
    return any(re.search(pattern, command) for pattern in patterns)


def _test_run_executed(record):
    """A zero exit code is not enough if a test runner discovered no tests."""
    if not record.get("success") or record.get("return_code") != 0:
        return False
    command = str(record.get("command", "")).lower()
    output = (str(record.get("stdout", "")) + "\n" + str(record.get("stderr", ""))).lower()
    if re.search(r"\b(?:pytest|unittest)\b", command):
        return bool(re.search(r"\bran\s+[1-9]\d*\s+tests?\b", output) or re.search(r"[1-9]\d*\s+passed\b", output))
    if re.search(r"\bnpm\s+(?:test|run\s+test)\b", command):
        return bool(re.search(r"\b[1-9]\d*\s+(?:passing|passed|tests? passed)\b", output))
    return False

def _has_passing_behavioral_tests(commands, changed_files=None):
    """Require passing tests plus a test file created/updated as part of this task."""
    changed_files = changed_files or []
    suffixes = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
    normalized = [str(path).replace(chr(92), "/").lower() for path in changed_files]
    changed_test_file = any(
        "/tests/" in f"/{path}" or "/test/" in f"/{path}"
        or (
            os.path.basename(path).startswith("test_")
            and os.path.splitext(path)[1] in suffixes
        )
        or any(os.path.basename(path).endswith("_test" + suffix) for suffix in suffixes)
        for path in normalized
    )
    if not changed_test_file:
        return False
    return any(_test_run_executed(item) for item in commands if isinstance(item, dict))


def _test_counts(command, stdout="", stderr=""):
    """Extract only counts actually printed by recognizable test runners."""
    output = str(stdout or "") + "\n" + str(stderr or "")
    command = str(command or "").lower()
    counts = {"discovered": None, "passed": None, "failed": None}
    if re.search(r"\bpytest\b", command):
        passed = re.search(r"(\d+)\s+passed\b", output)
        failed = re.search(r"(\d+)\s+failed\b", output)
        errors = re.search(r"(\d+)\s+error(?:s)?\b", output)
        if passed or failed or errors:
            counts["passed"] = int(passed.group(1)) if passed else 0
            counts["failed"] = (int(failed.group(1)) if failed else 0) + (int(errors.group(1)) if errors else 0)
            counts["discovered"] = counts["passed"] + counts["failed"]
    elif re.search(r"\b(?:unittest|discover)\b", command):
        ran = re.search(r"Ran\s+(\d+)\s+tests?\b", output, re.I)
        failures = re.search(r"failures=(\d+)", output, re.I)
        errors = re.search(r"errors=(\d+)", output, re.I)
        if ran:
            counts["discovered"] = int(ran.group(1))
            counts["failed"] = (int(failures.group(1)) if failures else 0) + (int(errors.group(1)) if errors else 0)
            counts["passed"] = max(0, counts["discovered"] - counts["failed"])
    return counts


def _expected_artifact_paths(task):
    """Infer explicit file/folder deliverables so premature 'done' is caught."""
    if not isinstance(task, str):
        return []

    folders = re.findall(
        r"\bfolder\s+(?:named|called|name|ka\s+naam)\s+[\x60'\"]?([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)",
        task,
        flags=re.IGNORECASE,
    )
    # Hinglish commonly puts the folder name after "folder" and before "banao".
    folders.extend(re.findall(
        r"\bfolder\s+[\x60'\"]?([A-Za-z0-9_-]+)[\x60'\"]?\s+(?:banao|banaye|bana\s+do|create\s+karo)",
        task,
        flags=re.IGNORECASE,
    ))
    # Also understand common Hinglish requests such as "folder banao gui".
    folders.extend(re.findall(
        r"\bfolder\s+(?:banao|banaye|bana\s+do|create\s+karo)\s+([A-Za-z0-9_-]+)",
        task,
        flags=re.IGNORECASE,
    ))
    folders.extend(re.findall(
        r"\b([A-Za-z0-9_-]+)\s+folder\s+(?:banao|banaye|bana\s+do)",
        task,
        flags=re.IGNORECASE,
    ))
    stop_words = {"aur", "and", "or", "usme", "isme", "then", "with", "code"}
    folders = list(dict.fromkeys(
        folder for folder in folders if folder.lower() not in stop_words
    ))
    # Capture explicitly mentioned source/config files even when the user phrases
    # the request in Hindi/Hinglish ("main.py mein code likho").
    file_paths = re.findall(
        r"((?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_-]+\."
        r"(?:py|pyw|js|jsx|ts|tsx|html|css|json|md|txt|java|cpp|c|cs|go|rs|php|sql|"
        r"yml|yaml|toml|sh|bat|ps1|env))\b",
        task,
        flags=re.IGNORECASE,
    )

    expected = list(folders)
    inside_folder = bool(
        re.search(
            r"\binside\s+(?:it|that folder|the folder)\b|"
            r"\b(folder|directory)\s+ke\s+andar\b|"
            r"\busme\b|\buske\s+andar\b|\bus\s+folder\s+ke\s+andar\b",
            task,
            re.IGNORECASE,
        )
    )
    for file_path in file_paths:
        normalized = file_path.replace("\\", "/")
        if inside_folder and folders and "/" not in normalized:
            normalized = folders[-1] + "/" + normalized
        if normalized not in expected:
            expected.append(normalized)

    # Infer concrete deliverables for common requests that do not name files.
    # Keep this list conservative: the recovery writer may only write these paths.
    lowered_task = task.lower()
    if not file_paths and not folders:
        if re.search(r"\bcalculator\b", lowered_task):
            expected.extend(["calculator.py", "test_calculator.py"])
        elif re.search(r"\b(?:todo|to-do|task manager)\b", lowered_task):
            expected.extend(["app.py", "test_app.py"])
        elif re.search(r"\b(?:create|build|make|implement|banao|bana do)\b", lowered_task) and re.search(r"\b(?:app|application|system|tool)\b", lowered_task):
            expected.extend(["app.py", "test_app.py"])

    # Only return relative paths; all actual operations remain workspace-scoped.
    return [path for path in expected if path and not os.path.isabs(path)]


class MultiModelAgent:
    """A single-agent, tool-driven coding loop with persistent short history."""

    def __init__(self, config, tools, tool_schemas, workspace, status=None, event_bus=None):
        self.config = config
        if self.config.get("execution_mode") == "offline":
            self.config.update(provider="ollama", local_only=True, hosted_api_enabled=False,
                               hosted_coder_escalation=False, provider_fallback_enabled=False)
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.workspace = workspace
        self.status_callback = status
        self.event_bus = event_bus
        self.models = config.get("models", {})
        self.max_total_workers = max(2, min(int(config.get("max_total_workers", 3)), 8))
        self.max_workers = max(1, min(
            int(config.get("max_parallel_tool_workers", config.get("max_parallel_workers", 2))),
            4, self.max_total_workers - 1,
        ))
        self.max_concurrent_llm_requests = max(1, min(
            int(config.get("max_concurrent_llm_requests", 1)), 2,
            self.max_total_workers - self.max_workers,
        ))
        self._llm_semaphore = threading.BoundedSemaphore(self.max_concurrent_llm_requests)
        self.max_steps = max(1, min(int(config.get("max_agent_steps", 32)), 80))
        self.session_history = []
        self.last_task_report = {}
        self._terminal_file_state = None
        self.workspace_context = WorkspaceContextManager(
            workspace, max_context_chars=int(config.get("workspace_context_chars", 6500))
        )
        memory_config = dict(config)
        memory_config.setdefault("persistent_memory_enabled", False)
        self.workspace_memory = WorkspaceMemory(workspace, self.workspace_context, memory_config)
        self.recent_tool_calls = []
        self.task_tool_events = []
        self.current_plan = []
        self.current_task_id = None
        self._context_dirty = True
        self.task_deadline_seconds = max(30, min(int(config.get("task_deadline_seconds", 300)), 1800))
        self._active_task_deadline = None

    def _record_tool_outcome(self, name, arguments, result):
        safe_arguments = {
            key: value for key, value in arguments.items()
            if key not in {"content", "old_text", "new_text"}
        }
        if "content" in arguments:
            safe_arguments["content_chars"] = len(str(arguments["content"]))
        if "old_text" in arguments or "new_text" in arguments:
            safe_arguments["edit_chars"] = {
                "old": len(str(arguments.get("old_text", ""))),
                "new": len(str(arguments.get("new_text", ""))),
            }
        if isinstance(result, dict):
            safe_result = {
                key: result.get(key) for key in (
                    "success", "error", "operation", "path", "return_code", "cwd",
                    "duration_seconds", "timed_out", "blocked", "file_state",
                ) if key in result
            }
            for key in ("stdout", "stderr"):
                if result.get(key):
                    safe_result[key] = str(result[key])[-1000:]
        else:
            safe_result = {"success": True, "lines": len(str(result).splitlines())} if isinstance(result, str) else {"success": True}
        record = {"tool": name, "arguments": safe_arguments, "result": safe_result}
        serialized = redact_secrets(json.dumps(record, ensure_ascii=False, default=str))
        self.recent_tool_calls.append(json.loads(serialized))
        self.recent_tool_calls = self.recent_tool_calls[-12:]
        training_record = json.loads(serialized)
        for key in ("content", "old_text", "new_text"):
            if key in arguments:
                training_record["arguments"][key] = redact_secrets(str(arguments[key]))[:2400]
        if isinstance(result, str):
            training_record["result"] = {"success": True, "content": redact_secrets(result)[:2400]}
        elif isinstance(result, dict) and isinstance(result.get("content"), str):
            training_record["result"]["content"] = redact_secrets(result["content"])[:2400]
        self.task_tool_events.append(training_record)
        self.task_tool_events = self.task_tool_events[-200:]
        if name in {"write_file", "edit_file"}:
            self.workspace_context.invalidate([str(arguments.get("path", ""))])
            self._context_dirty = True
        elif name in {"create_folder", "run_command"}:
            self.workspace_context.invalidate()
            self._context_dirty = True
        return result

    def _format_final_report(self, summary):
        """Append a factual execution report based only on recorded tool results."""
        report = self.last_task_report or {}
        changed = list(dict.fromkeys(report.get("changed_files", [])))
        commands = report.get("commands", [])
        checks = report.get("static_checks", [])

        lines = [str(summary or "").strip(), "", "Execution report:"]
        lines.append("Files changed: " + (", ".join(changed) if changed else "none recorded"))
        requests = report.get("llm_requests", [])
        attempts = report.get("llm_attempts", requests)
        tool_operations = report.get("tool_durations", [])
        lines.append(
            f"Execution counters: LLM calls {len(requests)}; provider request attempts {len(attempts)} "
            f"({sum(bool(item.get('success')) for item in attempts)} passed, "
            f"{sum(item.get('success') is False for item in attempts)} failed); "
            f"logical call results "
            f"({sum(bool(item.get('success')) for item in requests)} passed, "
            f"{sum(not item.get('success') for item in requests)} failed); "
            f"tool operations {len(tool_operations)} "
            f"({sum(bool(item.get('success')) for item in tool_operations)} passed, "
            f"{sum(not item.get('success') for item in tool_operations)} failed)."
        )
        if report.get("implementation_source"):
            lines.append("Implementation source: " + report["implementation_source"])

        latest_checks = {}
        for item in checks:
            latest_checks[item.get("path", "unknown")] = item
        if latest_checks:
            lines.append("Automatic file checks:")
            for path, item in latest_checks.items():
                state = "PASS" if item.get("success") else "FAIL"
                details = "; ".join(item.get("checks", []) + item.get("errors", []))
                lines.append(f"- {state} {path}" + (f" — {details}" if details else ""))

        if commands:
            lines.append("Commands executed:")
            for item in commands[-8:]:
                state = "PASS" if item.get("success") and item.get("return_code") == 0 else "FAIL"
                lines.append(f"- {state} [{item.get('return_code', 'unknown')}] {item.get('command', '')}")
                output = (str(item.get("stdout", "")) + "\n" + str(item.get("stderr", ""))).strip()
                if output:
                    compact_output = output[-1200:].replace("\n", " | ")
                    lines.append(f"  Actual output: {compact_output}")
                if item.get("duration_seconds") is not None:
                    lines.append(f"  Duration: {item['duration_seconds']:.2f}s; working directory: {item.get('cwd', self.workspace)}")
                if item.get("timed_out"):
                    lines.append("  Command terminated on timeout.")
                test_counts = item.get("test_counts", {})
                if test_counts.get("discovered") is not None:
                    lines.append(
                        f"  Tests: discovered {test_counts['discovered']}, passed {test_counts['passed']}, failed {test_counts['failed']}"
                    )

        meaningful = [item for item in commands if item.get("meaningful_check")]
        latest_meaningful_pass = bool(
            meaningful and meaningful[-1].get("success")
            and meaningful[-1].get("return_code") == 0
        )
        behavioral_tests_pass = _has_passing_behavioral_tests(commands, changed)
        report["behavioral_tests_pass"] = behavioral_tests_pass
        lines.append("Behavioral test gate: " + ("PASS — task-updated test file and passing test run recorded" if behavioral_tests_pass else "NOT PASSED — task-updated behavioral tests must be run successfully"))
        static_pass = all(item.get("success") for item in latest_checks.values())
        missing_artifacts = self._missing_expected_artifacts(report.get("task", ""))
        report["missing_artifacts"] = missing_artifacts
        spec_issues = self._task_completeness_issues() if report.get("specification") else []
        report["completeness_issues"] = spec_issues
        if report.get("deadline_reached") and not changed:
            lines.append("Verification status: BLOCKED — the task deadline expired before files were changed.")
        elif changed and latest_meaningful_pass and behavioral_tests_pass and static_pass and not missing_artifacts and not spec_issues:
            lines.append("Verification status: PASS — static checks passed and a relevant test runner discovered and passed task-specific tests.")
        elif changed:
            reasons = []
            if not latest_meaningful_pass:
                reasons.append("no successful relevant test/syntax/build command was recorded after the changes")
            if not static_pass:
                reasons.append("one or more latest automatic file checks failed")
            if not behavioral_tests_pass:
                reasons.append("no successful task-specific behavioral test run with at least one discovered test was recorded")
            if missing_artifacts:
                reasons.append("requested deliverables are still missing: " + ", ".join(missing_artifacts))
            lines.append("Verification status: PARTIAL — " + "; ".join(reasons) + ".")
        else:
            if missing_artifacts:
                lines.append("Verification status: FAIL — requested deliverables are missing: " + ", ".join(missing_artifacts) + ".")
            else:
                lines.append("Verification status: no file modifications were recorded; review the task summary above.")
        if report.get("deadline_reached") and not changed:
            report["verification_status"] = "BLOCKED"
        elif changed and latest_meaningful_pass and behavioral_tests_pass and static_pass and not missing_artifacts and not spec_issues:
            report["verification_status"] = "PASS"
        elif changed:
            report["verification_status"] = "PARTIAL"
        elif missing_artifacts:
            report["verification_status"] = "FAIL"
        else:
            report["verification_status"] = "NOT VERIFIED"
        if report.get("architect"):
            item = report["architect"]
            lines.append(f"Architect: {item.get('provider', 'none')} / {item.get('model', 'none')} ({item.get('status', 'unknown')}).")
            if item.get("usage"):
                lines.append("Architect usage: " + json.dumps(item["usage"], ensure_ascii=False))
            if item.get("error"):
                lines.append("Architect failure detail: " + redact_secrets(str(item["error"]))[:900])
        if report.get("coder"):
            item = report["coder"]
            lines.append(f"Coder: {item.get('provider', 'unknown')} / {item.get('model', 'unknown')}.")
        if report.get("escalations"):
            lines.append(f"Hosted coder escalations: {len(report['escalations'])}.")
            for item in report["escalations"]:
                if item.get("error"):
                    lines.append(f"- {item.get('provider', 'provider')} {item.get('status', 'FAILED')}: {redact_secrets(str(item['error']))[:900]}")
        if report.get("initial_error"):
            lines.append("Initial execution error: " + redact_secrets(str(report["initial_error"]))[:1200])
        if report.get("usage"):
            lines.append("Coder usage: " + json.dumps(report["usage"], ensure_ascii=False))
        failed_requests = [item for item in report.get("llm_requests", []) if not item.get("success")]
        for item in failed_requests[-5:]:
            lines.append(
                f"Model request failed: {item.get('provider', 'unknown')} / {item.get('model', 'unknown')} "
                f"during {item.get('phase', 'task')} ({item.get('duration_seconds', 0):.1f}s): "
                + redact_secrets(str(item.get("error", "No error detail returned.")))[:900]
            )
        lines.extend("- Acceptance check FAIL: " + issue for issue in spec_issues)
        if not self.config.get("dashboard_show_technical", False):
            # Human-readable summary by default; verbose telemetry stays opt-in.
            concise = [str(summary or "").strip()]
            status = report.get("verification_status", "NOT VERIFIED")
            concise.append(f"\nStatus: {status}")
            if changed:
                concise.append("Files updated: " + ", ".join(changed))
            passing_tests = [
                item.get("test_counts", {})
                for item in commands
                if item.get("success") and item.get("test_counts", {}).get("discovered") is not None
            ]
            if passing_tests:
                latest_tests = passing_tests[-1]
                concise.append(f"Tests: {latest_tests.get('passed', 0)}/{latest_tests.get('discovered', 0)} passed")
            elif status != "PASS":
                concise.append("Tests: not confirmed as passing")
            if status != "PASS":
                if missing_artifacts:
                    concise.append("Missing files: " + ", ".join(missing_artifacts))
                if report.get("initial_error"):
                    concise.append("Issue: " + redact_secrets(str(report["initial_error"]))[:350])
                failed_commands = [
                    item for item in commands
                    if not item.get("success") or item.get("return_code") != 0
                ]
                if failed_commands:
                    last_failure = failed_commands[-1]
                    detail = (str(last_failure.get("stderr", "")) or str(last_failure.get("stdout", ""))).strip()
                    if detail:
                        concise.append("Last check error: " + redact_secrets(detail[-350:]).replace("\n", " "))
            return redact_secrets("\n".join(line for line in concise if line))

        return redact_secrets("\n".join(line for line in lines if line is not None))

    def _missing_expected_artifacts(self, task_context):
        """Return explicitly requested relative paths that do not exist yet."""
        missing = []
        root = os.path.realpath(self.workspace)
        specification = (self.last_task_report or {}).get("specification") or {}
        expected = [] if specification else list(_expected_artifact_paths(task_context))
        for item in specification.get("expected_files", []):
            path = item.get("path") if isinstance(item, dict) else None
            if path and path not in expected:
                expected.append(path)
        for artifact in expected:
            full_path = os.path.realpath(os.path.join(root, artifact))
            try:
                if os.path.commonpath([root, full_path]) != root:
                    continue
            except ValueError:
                continue
            if not os.path.exists(full_path):
                missing.append(artifact)
        return missing

    def _task_expected_paths(self, task_context):
        """Use the current validated contract when present; only guess paths without one."""
        specification = (self.last_task_report or {}).get("specification")
        if isinstance(specification, dict):
            return [item["path"] for item in specification.get("expected_files", [])
                    if isinstance(item, dict) and item.get("path")]
        return list(_expected_artifact_paths(task_context))

    def _task_completeness_issues(self):
        specification = (self.last_task_report or {}).get("specification")
        if not specification:
            return []
        issues = completeness_issues(self.workspace, specification)
        if specification.get("test_commands") and not any(
            _test_run_executed(record) for record in self.last_task_report.get("commands", [])
        ):
            issues.append("Required automated tests were not confirmed as executed with at least one discovered test")
        return issues

    def _recover_car_game_after_provider_failure(self, task, specification, error):
        """Build and verify a narrowly supported Tkinter car game through real tools."""
        if not supports_car_game_recovery(task):
            return None
        # A small-model specification may contain stale paths from the previous task
        # (e.g. calculator/main.py). The user explicitly requested a game, so the
        # deterministic recovery owns its known file layout instead of trusting those paths.
        if "write_file" not in self.tools or "run_command" not in self.tools:
            return None

        self.show_status("Game acceptance failed; building a playable Python GUI game locally")
        self.emit_event("workflow.phase", phase="implementing", state="active",
                        message="Writing and testing a standard-library Python GUI game after acceptance checks failed.")
        report = self.last_task_report
        report["implementation_source"] = "deterministic local Python GUI game recovery after provider or acceptance-check failure"
        report["provider_failure"] = redact_secrets(str(error))[:1200]
        report["recovery_status"] = "STARTED"
        recovery_files = car_game_files()
        if re.search(r"\b(folder|directory)\b", str(task or ""), re.IGNORECASE):
            recovery_files["game_project/README.md"] = (
                "# Python GUI Game Project\n\n"
                "This folder marks the generated game project. The playable entry point is `../main.py`.\n\n"
                "Run from the workspace root with `python main.py`.\n"
                "Controls: Left/Right arrows or A/D; press R to restart after a crash.\n"
            )
        for path, content in recovery_files.items():
            result = self.execute_tool("write_file", {"path": path, "content": content})
            if not isinstance(result, dict) or result.get("success") is not True:
                report["recovery_status"] = "FAILED"
                report["initial_error"] = "Recovery could not write " + path + ": " + str(
                    result.get("error", "unknown file-tool error") if isinstance(result, dict) else result
                )
                report["verification_status"] = "FAIL"
                return self._format_final_report("The game failed acceptance checks and the local recovery could not write all files.")
            if path not in report.setdefault("changed_files", []):
                report["changed_files"].append(path)
            report.setdefault("static_checks", []).append(verify_changed_file(self.workspace, path))

        command_list = [
            f'"{sys.executable}" -m py_compile main.py game_logic.py tests/test_game_logic.py',
            f'"{sys.executable}" -m unittest discover -s tests -p "test_game_logic.py" -v',
        ]
        all_passed = True
        for command in command_list:
            result = self.execute_tool("run_command", {"command": command})
            result = result if isinstance(result, dict) else {"success": False, "stderr": str(result)}
            record = {
                "command": command,
                "cwd": result.get("cwd", self.workspace),
                "duration_seconds": result.get("duration_seconds"),
                "timed_out": result.get("timed_out", False),
                "return_code": result.get("return_code"),
                "success": result.get("success") is True and result.get("return_code") == 0,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "meaningful_check": True,
            }
            record["test_counts"] = _test_counts(command, record["stdout"], record["stderr"])
            record["executed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            report.setdefault("commands", []).append(record)
            if command.endswith("-v") or "unittest discover" in command:
                report["test_results"] = record["test_counts"]
            self.emit_event("verification", message=f"{command} — {'PASS' if record['success'] else 'FAIL'}")
            all_passed = all_passed and record["success"]
        report["recovery_status"] = "PASS" if all_passed else "FAILED"
        report["verification_status"] = "PASS" if all_passed else "FAIL"
        if not all_passed:
            report["initial_error"] = "Local recovery files were written, but at least one verification command failed."
        result = self._format_final_report(
            "The requested game did not pass the model-generated acceptance checks. A deterministic Tkinter game recovery created the implementation and ran its checks."
            if all_passed else
            "The game recovery wrote files, but verification failed; the task is not complete."
        )
        return result

    def _recover_calculator_gui_task(self, task):
        """Build and verify a deterministic calculator in the requested project layout."""
        report = self.last_task_report
        folder_match = re.search(r"(?:folder|directory|project)\s+(?:named\s+|called\s+)?([a-zA-Z][a-zA-Z0-9_-]*)", str(task or ""), re.IGNORECASE)
        folder = folder_match.group(1) if folder_match and folder_match.group(1).lower() in {"calculator", "calc"} else "calculator"
        files = calculator_project_files(folder)
        report.update({
            "task": task, "changed_files": [], "commands": [], "static_checks": [], "specification": None,
            "architect": {"provider": "none", "model": "none", "status": "SKIPPED (deterministic calculator task)"},
            "coder": {"provider": "local recovery", "model": "Python standard library"},
            "implementation_source": "deterministic folder-aware Tkinter calculator recovery",
        })
        self.show_status("Building the calculator project and writing its exact entry-point paths")
        for path, content in files.items():
            written = self.execute_tool("write_file", {"path": path, "content": content})
            if not isinstance(written, dict) or written.get("success") is not True:
                report["initial_error"] = f"Could not write {path}: " + str(written.get("error", "unknown file tool error") if isinstance(written, dict) else written)
                report["verification_status"] = "FAIL"
                return self._format_final_report("Calculator recovery failed while writing source files.")
            if path not in report["changed_files"]:
                report["changed_files"].append(path)
            report["static_checks"].append(verify_changed_file(self.workspace, path))
        commands = [
            f'"{sys.executable}" -m py_compile main.py "{folder}/main.py" "{folder}/calculator.py" "{folder}/test_calculator.py"',
            f'"{sys.executable}" -m unittest discover -s "{folder}" -p "test_calculator.py" -v',
        ]
        for command in commands:
            outcome = self.execute_tool("run_command", {"command": command})
            outcome = outcome if isinstance(outcome, dict) else {"success": False, "stderr": str(outcome)}
            record = {"command": command, "cwd": outcome.get("cwd", self.workspace), "duration_seconds": outcome.get("duration_seconds"), "timed_out": outcome.get("timed_out", False), "return_code": outcome.get("return_code"), "success": outcome.get("success") is True and outcome.get("return_code") == 0, "stdout": outcome.get("stdout", ""), "stderr": outcome.get("stderr", ""), "meaningful_check": True}
            record["test_counts"] = _test_counts(command, record["stdout"], record["stderr"])
            record["executed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            report["commands"].append(record)
            self.emit_event("verification", message=f"{command} — {'PASS' if record['success'] else 'FAIL'}")
        checks_pass = all(item.get("success") for item in report["static_checks"])
        command_pass = bool(report["commands"]) and all(item.get("success") and item.get("return_code") == 0 for item in report["commands"])
        test_recorded = any(_test_run_executed(item) for item in report["commands"])
        report["verification_status"] = "PASS" if checks_pass and command_pass and test_recorded else "PARTIAL"
        report["behavioral_tests_pass"] = test_recorded
        if not command_pass or not test_recorded:
            report["initial_error"] = "Calculator files were generated, but compile/test verification failed or discovered no tests. Review command output below."
        summary = (f"Created the calculator project in {folder}/ with a root main.py launcher, GUI arithmetic operations, input validation, clear button, and unit tests." if report["verification_status"] == "PASS" else "Created calculator files, but verification did not fully pass; see command output.")
        return self._format_final_report(summary)

    def _generate_missing_files_locally(self, client, task_context, expected_paths):
        """Escape a stuck tool loop by asking the local model for file contents without tools."""
        file_paths = [
            path for path in expected_paths
            if os.path.splitext(path)[1].lower() in {
                ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".html", ".css",
                ".json", ".md", ".txt", ".java", ".cpp", ".c", ".cs", ".go",
                ".rs", ".php", ".sql", ".yml", ".yaml", ".toml", ".sh",
            }
        ]
        if not file_paths:
            return False

        last_error = ""
        for attempt in range(2):
            self.show_status("Local model stalled; generating required files directly")
            prompt = (
                "You are recovering a stalled local coding agent. Return ONLY one valid JSON object "
                "with this exact shape: {\"files\": {\"relative/path.ext\": \"complete file content\", ...}}. "
                "No Markdown fences, no commentary, and no tool-call JSON.\n"
                "Complete the user's entire task in the listed files. Write substantial, runnable implementation code, "
                "not placeholders, empty functions, TODOs, or comments instead of code. Use sensible defaults and include "
                "the most useful practical features implied by the task. Add meaningful tests for core behavior. "
                "Every required path must have complete content. Do not return folder names as files.\n"
                f"Workspace root: {self.workspace}\n"
                f"Required file paths (use these exact keys): {json.dumps(file_paths, ensure_ascii=False)}\n"
                f"Current project context and prior tool outcomes:\n{self._workspace_context_text(task_context, previous_report=self.last_task_report, specification=(self.last_task_report or {}).get('specification'))}\n"
                f"Original task:\n{redact_secrets(task_context)}\n"
                + (f"Previous verification failure to fix:\n{last_error}\n" if last_error else "")
            )
            try:
                response = self._chat_with_events(
                    client,
                    [
                        {
                            "role": "system",
                            "content": (
                                "Generate complete source files as a JSON files map. "
                                "The response is consumed by a program, so valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tools=None,
                    phase="recovery",
                )
                content = getattr(response.choices[0].message, "content", "") or ""
                payload = _extract_json_object(content)
                generated = payload.get("files") if isinstance(payload, dict) else None
                if not isinstance(generated, dict):
                    last_error = "Model did not return a JSON object with a files map."
                    continue

                allowed = {os.path.normcase(os.path.normpath(path)): path for path in file_paths}
                wrote_any = False
                written_paths = set()
                for path, file_content in generated.items():
                    if not isinstance(path, str) or not isinstance(file_content, str):
                        continue
                    normalized = os.path.normcase(os.path.normpath(path.replace("\\", os.sep)))
                    canonical = allowed.get(normalized)
                    if not canonical:
                        continue
                    if not file_content.strip():
                        last_error = f"Model returned empty source for {canonical}."
                        continue
                    result = self.execute_tool("write_file", {
                        "path": canonical,
                        "content": file_content,
                    })
                    if not isinstance(result, dict) or result.get("success") is not True:
                        last_error = f"Could not write {canonical}: {result}"
                        continue
                    wrote_any = True
                    written_paths.add(canonical)
                    if canonical not in self.last_task_report["changed_files"]:
                        self.last_task_report["changed_files"].append(canonical)

                not_generated = sorted(set(file_paths) - written_paths)
                if not_generated:
                    last_error = "Model omitted required implementation files: " + ", ".join(not_generated)
                    continue

                # Re-read and validate every file from this recovery attempt. Existence
                # alone is not enough: a zero-byte source file must never count as done.
                checks = [verify_changed_file(self.workspace, path) for path in file_paths]
                self.last_task_report["static_checks"].extend(checks)
                failed_checks = [check for check in checks if not check.get("success")]
                if failed_checks:
                    last_error = "File validation failed: " + "; ".join(
                        f"{item.get('path')}: {', '.join(item.get('errors', []))}"
                        for item in failed_checks
                    )
                    continue

                missing = self._missing_expected_artifacts(task_context)
                if missing:
                    last_error = "Still missing requested artifacts: " + ", ".join(missing)
                    continue

                python_files = [path for path in file_paths if path.lower().endswith((".py", ".pyw"))]
                test_files = [path for path in python_files if os.path.basename(path).lower().startswith("test_")]
                if test_files:
                    test_dir = os.path.dirname(test_files[0]).replace("\\", "/") or "."
                    command = f'"{sys.executable}" -m unittest discover -s "{test_dir}" -v'
                elif python_files:
                    targets = " ".join(f'"{path}"' for path in python_files)
                    command = f'"{sys.executable}" -m py_compile {targets}'
                elif any(path.lower().endswith((".js", ".mjs", ".cjs")) for path in file_paths):
                    command = "node --check " + " ".join(
                        f'"{path}"' for path in file_paths if path.lower().endswith((".js", ".mjs", ".cjs"))
                    )
                else:
                    command = ""

                if command:
                    result = self.execute_tool("run_command", {"command": command})
                    record = {
                        "command": command,
                        "return_code": result.get("return_code") if isinstance(result, dict) else None,
                        "success": isinstance(result, dict) and result.get("success") is True,
                        "stdout": result.get("stdout", "") if isinstance(result, dict) else "",
                        "stderr": result.get("stderr", "") if isinstance(result, dict) else "",
                        "meaningful_check": _is_meaningful_verification_command(command),
                    }
                    self.last_task_report["commands"].append(record)
                    record["executed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    self.workspace_memory.record_command(record)
                    output = (
                        str(result.get("stdout", "") if isinstance(result, dict) else result)
                        + "\n"
                        + str(result.get("stderr", "") if isinstance(result, dict) else "")
                    )
                    tests_ran = not test_files or bool(
                        re.search(r"Ran\s+[1-9]\d*\s+tests?\b", output, re.IGNORECASE)
                    )
                    if record["success"] and record["return_code"] == 0 and record["meaningful_check"] and tests_ran:
                        complete = not self._missing_expected_artifacts(task_context)
                        if self.last_task_report.get("specification"):
                            complete = complete and not self._task_completeness_issues()
                        return complete
                    if test_files and not tests_ran:
                        last_error = "Test command returned zero but did not report running any tests. Check discovery/import paths."
                    else:
                        last_error = f"Command failed (return code {record['return_code']}): {command}"
                    last_error += "\n" + output
                    continue

                complete = not self._missing_expected_artifacts(task_context)
                if self.last_task_report.get("specification"):
                    complete = complete and not self._task_completeness_issues()
                return complete
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"

        return False

    def show_status(self, message):
        if self.status_callback:
            self.status_callback(message)

    def emit_event(self, event, **data):
        if self.event_bus:
            self.event_bus.emit(event, **data)

    def _chat_with_events(self, client, messages, tools=None, provider=None, phase="model"):
        started = time.monotonic()
        self.emit_event("model.request", provider=provider or self.config.get("provider", "configured"),
                        model=getattr(client, "model", "configured"), phase=phase)
        timeout = max(1, int(self.config.get("llm_slot_timeout_seconds", self.config.get("request_timeout_seconds", 90))))
        acquired = self._llm_semaphore.acquire(timeout=timeout)
        if not acquired:
            message = "Timed out waiting for an LLM concurrency slot"
            self.last_task_report.setdefault("llm_requests", []).append({
                "provider": provider or self.config.get("provider", "configured"),
                "model": getattr(client, "model", "configured"), "phase": phase,
                "duration_seconds": time.monotonic() - started, "success": False, "error": message,
            })
            self.emit_event("model.response", duration=time.monotonic() - started,
                            success=False, phase=phase, error=message)
            raise TimeoutError(message)
        attempt_record = {
            "provider": provider or self.config.get("provider", "configured"),
            "model": getattr(client, "model", "configured"),
            "phase": phase,
            "attempt": 1,
            "success": None,
        }
        self.last_task_report.setdefault("llm_attempts", []).append(attempt_record)
        self._current_model = attempt_record["model"]
        self._current_phase = phase
        try:
            response = client.chat(messages, tools=tools)
        except Exception as error:
            self.last_task_report["llm_attempts"][-1]["success"] = False
            duration = time.monotonic() - started
            error_message = safe_error(error, ("GROQ_API_KEY", "NVIDIA_API_KEY"))
            self.last_task_report.setdefault("llm_requests", []).append({
                "provider": provider or self.config.get("provider", "configured"),
                "model": getattr(client, "model", "configured"), "phase": phase,
                "duration_seconds": duration, "success": False, "error": error_message,
            })
            self.emit_event("model.response", duration=time.monotonic() - started,
                            success=False, phase=phase, error=error_message)
            raise
        finally:
            self._llm_semaphore.release()
        duration = time.monotonic() - started
        self.last_task_report["llm_attempts"][-1]["success"] = True
        self.last_task_report.setdefault("llm_requests", []).append({
            "provider": provider or self.config.get("provider", "configured"),
            "model": getattr(client, "model", "configured"), "phase": phase,
            "duration_seconds": duration, "success": True,
        })
        self.emit_event("model.response", duration=time.monotonic() - started,
                        success=True, phase=phase)
        return response

    def _record_provider_retry(self, provider, attempt, attempts, delay, error):
        requests = self.last_task_report.setdefault("llm_attempts", [])
        if requests and requests[-1].get("success") is None:
            requests[-1]["success"] = False
        requests.append({
            "provider": provider,
            "model": getattr(self, "_current_model", "configured"),
            "phase": getattr(self, "_current_phase", "model"),
            "attempt": int(attempt) + 1,
            "success": None,
        })
        self.emit_event("provider.retry", provider=provider.upper(), attempt=attempt,
                        attempts=attempts, delay=delay, error=safe_error(error))

    def get_client(self, role="coder", provider=None, model=None):
        provider = str(provider or self.config.get("provider", "ollama" if self.config.get("execution_mode") == "offline" else "groq")).strip().lower()
        if self.config.get("execution_mode") == "offline" and provider not in {"ollama", "local"}:
            raise RuntimeError(f"Offline mode blocks hosted provider '{provider}'. Select online mode to use hosted APIs.")
        common = {
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 8192),
        }

        if provider in {"ollama", "local"}:
            return OllamaClient(
                model=self.config.get("local_model", "qwen2.5-coder:1.5b"),
                base_url=self.config.get(
                    "ollama_base_url", "http://localhost:11434/v1"
                ),
                force_json_tools=self.config.get("ollama_force_json_tools", True),
                timeout=self.config.get("request_timeout_seconds", 90),
                **common,
            )

        if provider in {"nvidia", "nvidia_nim"}:
            retry = self.config.get("retry", {})
            role_model = self.config.get(f"nvidia_{role}_model")
            if role == "planner":
                role_model = role_model or self.config.get("nvidia_architect_model")
            configured_model = (model
                                or (self.config.get("selected_model") if role == "coder" else None)
                                or role_model
                                or self.config.get("nvidia_model", "qwen/qwen2.5-coder-32b-instruct"))
            return NvidiaClient(
                model=configured_model,
                base_url=self.config.get(
                    "nvidia_base_url", "https://integrate.api.nvidia.com/v1"
                ),
                timeout=self.config.get(
                    "nvidia_request_timeout_seconds",
                    self.config.get("request_timeout_seconds", 90),
                ),
                **common,
                max_attempts=retry.get("max_attempts", 3),
                base_delay=retry.get("base_delay", 1.5),
                on_retry=lambda attempt, attempts, delay, error: self._record_provider_retry(
                    "nvidia", attempt, attempts, delay, error,
                ),
            )

        if provider != "groq":
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'nvidia', 'groq', or 'ollama'."
            )

        model = model or self.models.get(role) or self.models.get("coder")
        if not model:
            raise KeyError(f"No model configured for role: {role}")

        retry = self.config.get("retry", {})
        return GroqClient(
            model=model,
            timeout=self.config.get("request_timeout_seconds", 90),
            **common,
            max_attempts=retry.get("max_attempts", 3),
            base_delay=retry.get("base_delay", 1.5),
            on_retry=lambda attempt, attempts, delay, error: self._record_provider_retry(
                "groq", attempt, attempts, delay, error,
            ),
        )

    def classify_task(self, task):
        text = task.lower()
        if any(word in text for word in (
            "build", "create", "implement", "application", "website", "game",
            "backend", "frontend", "database", "authentication", "refactor",
            "deploy", "integrate", "banao", "bana do", "project", "full app",
        )):
            return "COMPLEX"
        if any(word in text for word in (
            "fix", "debug", "modify", "edit", "add", "feature", "bug", "error",
            "api", "component", "function",
        )):
            return "MEDIUM"
        return "SIMPLE"

    @staticmethod
    def _has_provider_key(provider):
        key_name = "GROQ_API_KEY" if provider == "groq" else "NVIDIA_API_KEY"
        value = os.getenv(key_name, "").strip()
        return bool(value and not re.match(r"(?i)^(your_|replace|put[_ -]?|<|\$\{)", value))

    def _hosted_candidates(self, role, configured="auto"):
        if self.config.get("execution_mode") == "offline" or self.config.get("local_only") or self.config.get("hosted_api_enabled") is False:
            return []
        configured = str(configured or "auto").strip().lower()
        valid = {"nvidia", "nvidia_nim", "groq"}
        if configured == "auto":
            current = str(self.config.get("provider", "ollama")).lower()
            preferred = str(self.config.get("hosted_provider_preference", "auto")).lower()
            selected = current if current in valid else preferred
            order = [selected] if selected in valid else []
        elif configured in valid:
            order = [configured]
        else:
            return []
        result = []
        for provider in order:
            if provider == "nvidia_nim":
                provider = "nvidia"
            if provider in result or not self._has_provider_key(provider):
                continue
            if provider == "groq":
                model = self.config.get("architect_model") if role == "planner" else self.config.get("coder_model")
                model = model or self.models.get(role) or self.models.get("coder")
            else:
                model = self.config.get(f"nvidia_{role}_model")
                if role == "planner":
                    model = model or self.config.get("nvidia_architect_model")
                model = model or self.config.get("nvidia_model")
            if model:
                result.append(provider)
        return result

    def _model_for(self, provider, role):
        if provider == "groq":
            return (self.config.get("architect_model") if role == "planner" else self.config.get("coder_model")) or self.models.get(role) or self.models.get("coder")
        model = self.config.get(f"nvidia_{role}_model")
        if role == "planner":
            model = model or self.config.get("nvidia_architect_model")
        return model or self.config.get("nvidia_model")

    @staticmethod
    def _response_text(response):
        try:
            return (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, TypeError):
            return ""

    @staticmethod
    def _usage_metadata(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        values = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if isinstance(value, (int, float)):
                values[name] = value
        return values or None

    def _workspace_context_text(self, task, previous_report=None, specification=None):
        memory_state = self.workspace_memory.refresh()
        context = self.workspace_context.build_context(
            task=task,
            previous_report=previous_report or {},
            recent_history=self.session_history,
            recent_tool_calls=self.recent_tool_calls,
            specification=specification,
            refresh=False,
        )
        memory_text = self.workspace_memory.context_for(
            task, max_chars=int(self.config.get("workspace_memory_context_chars", 1200)), refresh=False
        )
        if self.last_task_report and self.last_task_report.get("task_id") == self.current_task_id:
            self.last_task_report["context_snapshot"] = redact_secrets(context + "\n\n" + memory_text)[:18000]
            self.last_task_report["memory_context"] = {
                "workspace_id": self.workspace_memory.workspace_id,
                "status": self.workspace_memory.status,
                "indexed_files": memory_state.get("indexed_files", 0),
                "retrieved_paths": list(self.workspace_memory.retrieved_paths),
                "project_fingerprint": self.workspace_memory._data["manifest"].get("project_fingerprint"),
            }
        self.emit_event("workspace.memory", **self.workspace_memory.dashboard_state())
        return context + "\n\n" + memory_text

    def _trajectory_source_changes(self, report):
        if not self.workspace_memory.enabled:
            return {}
        changes, total = {}, 0
        root = os.path.realpath(self.workspace)
        for relative in report.get("changed_files", [])[:12]:
            normalized = str(relative).replace("\\", "/")
            if self.workspace_context._excluded(normalized):
                continue
            full = os.path.realpath(os.path.join(root, normalized))
            try:
                if os.path.commonpath([root, full]) != root:
                    continue
                size = os.path.getsize(full)
                if size > 12000 or total + size > 48000:
                    continue
                with open(full, "r", encoding="utf-8", errors="replace") as source:
                    content = redact_secrets(source.read(12000))
                changes[normalized] = content
                total += len(content.encode("utf-8"))
            except (OSError, ValueError):
                continue
        return changes

    def _architect_task(self, task, previous_report=None):
        """Use one configured hosted model to create a validated task contract."""
        fallback = fallback_specification(task, workspace_files=self.workspace_context.files())
        metadata = {"provider": "none", "model": "none", "status": "NOT RUN", "error": None}
        if not self.config.get("task_architect_enabled", True) or self.classify_task(task) != "COMPLEX":
            metadata["status"] = "SKIPPED (simple task)"
            return fallback, metadata
        providers = self._hosted_candidates("planner", self.config.get("architect_provider", "auto"))
        if not providers:
            metadata["status"] = "NOT RUN (local-only or no configured hosted credentials/model)"
            return fallback, metadata

        schema = {
            "task_type": "string", "project_name": "string", "intended_outcome": "string",
            "language": "string", "framework": "string", "features": ["string"],
            "expected_files": [{"path": "relative/path", "responsibility": "string"}],
            "dependencies": ["string"], "run_commands": ["string"],
            "test_commands": ["string"], "acceptance_criteria": ["string"],
            "assumptions": ["string"], "definition_of_done": "string",
        }
        prompt = (
            "Analyze the user's actual software task and return exactly one JSON object matching this schema. "
            "Choose sensible defaults rather than asking nonessential questions. Infer the project type; do not force a game. "
            "For a game, specify playable mechanics, input, progression/scoring, end/restart behavior, files, and tests. "
            "List only concrete, necessary files using safe workspace-relative paths. Avoid credentials and secrets. "
            "Do not claim implementation or test results.\nSCHEMA:\n"
            + json.dumps(schema, ensure_ascii=False)
            + "\nUSER TASK:\n" + redact_secrets(task)
            + "\n\nCOMPACT LIVE WORKSPACE CONTEXT:\n"
            + self._workspace_context_text(task, previous_report=previous_report)
        )
        errors = []
        for provider in providers:
            model = self._model_for(provider, "planner")
            metadata.update({"provider": provider, "model": model})
            try:
                client = self.get_client("planner", provider=provider, model=model)
                response = self._chat_with_events(
                    client,
                    [{"role": "system", "content": "You are a software task architect. Return machine-readable JSON only."}, {"role": "user", "content": prompt}],
                    tools=None, provider=provider, phase="planning",
                )
                specification = parse_specification(self._response_text(response))
                return specification, {"provider": provider, "model": getattr(client, "model", model), "status": "PASS", "error": None, "usage": self._usage_metadata(response)}
            except Exception as error:
                errors.append(f"{provider}: {type(error).__name__}: {redact_secrets(str(error))[:600]}")
        metadata["status"] = "FAILED; deterministic specification used"
        metadata["error"] = "; ".join(errors)
        return fallback, metadata

    def _escalation_provider(self, original_provider=None):
        configured = self.config.get("escalation_provider", "auto")
        candidates = self._hosted_candidates("coder", configured)
        if configured == "auto" and original_provider in {"groq", "nvidia", "nvidia_nim"}:
            alternatives = [item for item in candidates if item != original_provider]
            if alternatives:
                return alternatives[0]
        return candidates[0] if candidates else None

    def execute_tool(self, name, arguments):
        started = time.monotonic()
        safe_arguments = {key: value for key, value in arguments.items()
                          if key not in {"content", "old_text", "new_text"}} if isinstance(arguments, dict) else {}
        self.emit_event("tool.started", tool=name, label=self._status_for_tool(name, arguments),
                        arguments=safe_arguments)

        def finish(result):
            self.last_task_report.setdefault("tool_durations", []).append({
                "tool": name, "duration_seconds": time.monotonic() - started,
                "success": not (isinstance(result, dict) and result.get("success") is False),
            })
            safe_result = result if isinstance(result, dict) else {"success": True, "summary": str(result)[:500]}
            safe_result = {
                key: (redact_secrets(str(value))[-500:] if isinstance(value, str) else value)
                for key, value in safe_result.items()
                if key in {"success", "error", "return_code", "cwd", "duration_seconds", "timed_out", "stdout", "stderr", "summary"}
            }
            self.emit_event("tool.result", tool=name, command=redact_secrets(str(arguments.get("command", "")))[:500],
                            result=safe_result, duration=time.monotonic() - started)
            return self._record_tool_outcome(name, arguments if isinstance(arguments, dict) else {}, result)

        if name not in self.tools:
            result = {"success": False, "error": f"Unknown tool: {name}"}
            self.emit_event("tool.finished", tool=name, duration=time.monotonic() - started, success=False)
            return finish(result)
        if not isinstance(arguments, dict):
            result = {"success": False, "error": "Tool arguments must be a JSON object."}
            self.emit_event("tool.finished", tool=name, duration=time.monotonic() - started, success=False)
            return finish(result)
        if name == "read_file":
            path_name = os.path.basename(str(arguments.get("path", ""))).lower()
            if path_name == ".env" or path_name.startswith(".env.") or path_name.endswith((".pem", ".key")) or path_name in {"id_rsa", "id_ed25519", "credentials.json"}:
                result = {"success": False, "error": "Reading credential and private-key files is blocked."}
                self.emit_event("tool.finished", tool=name, duration=time.monotonic() - started, success=False)
                return finish(result)

        command_snapshot = self._workspace_file_snapshot() if name == "run_command" else None
        path = arguments.get("path") if name in {"read_file", "write_file", "edit_file"} else None
        full_path = os.path.abspath(os.path.join(self.workspace, path)) if path else None
        try:
            if full_path and os.path.commonpath([os.path.abspath(self.workspace), full_path]) != os.path.abspath(self.workspace):
                full_path = None
        except ValueError:
            full_path = None
        existed_before = bool(full_path and os.path.isfile(full_path))
        before = None
        if name in {"write_file", "edit_file"} and existed_before:
            try:
                if os.path.getsize(full_path) <= 2_000_000:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as file:
                        before = file.read()
            except OSError:
                before = None

        try:
            call_arguments = dict(arguments)
            if name == "run_command" and self.config.get("execution_mode") == "offline":
                call_arguments["offline"] = True
            result = self.tools[name](self.workspace, **call_arguments)
        except Exception as error:
            result = {"success": False, "error": f"{type(error).__name__}: {redact_secrets(str(error))}"}
            self.emit_event("tool.finished", tool=name, duration=time.monotonic() - started, success=False)
            return finish(result)

        if path:
            relative = os.path.relpath(full_path, os.path.abspath(self.workspace)).replace("\\", "/") if full_path else str(path).replace("\\", "/")
            if name == "read_file" and isinstance(result, str):
                self.emit_event("file.activity", action="read", path=relative, lines=len(result.splitlines()))
            elif name in {"write_file", "edit_file"} and isinstance(result, dict) and result.get("success"):
                after = None
                try:
                    if os.path.getsize(full_path) <= 2_000_000:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as file:
                            after = file.read()
                except (OSError, TypeError):
                    pass
                line_count = len(after.splitlines()) if after is not None else (result.get("file_state") or {}).get("lines")
                action = "created" if not existed_before else "modified"
                if action == "created":
                    added, removed = line_count, 0
                elif before is not None and after is not None:
                    added, removed = _dashboard_line_delta(before, after)
                else:
                    added, removed = None, None
                self.emit_event("file.activity", action=action, path=relative,
                                lines=line_count, added=added, removed=removed)
        if name == "create_folder" and isinstance(result, dict) and result.get("success"):
            folder = str(arguments.get("path", "")).replace("\\", "/").strip("/")
            self.emit_event("file.activity", action="folder_created", path=folder)
        if command_snapshot is not None:
            self._emit_command_file_changes(command_snapshot)

        def sanitize(value):
            if isinstance(value, str):
                return redact_secrets(value)
            if isinstance(value, list):
                return [sanitize(item) for item in value]
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            return value

        result = sanitize(result)

        # Bound tool output so a large repository or command output cannot flood context.
        try:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            serialized = str(result)

        if len(serialized) > MAX_TOOL_RESULT_CHARS:
            serialized = serialized[:MAX_TOOL_RESULT_CHARS] + "\n...[tool output truncated]"
            self.emit_event("tool.finished", tool=name, duration=time.monotonic() - started, success=True)
            return finish({"success": True, "truncated": True, "output": serialized})
        self.emit_event("tool.finished", tool=name, duration=time.monotonic() - started,
                        success=not (isinstance(result, dict) and result.get("success") is False))
        return finish(result)

    def _workspace_file_snapshot(self):
        """Cache signatures and line counts within a bounded per-scan read budget."""
        snapshot = {}
        scan_budget = MAX_COMMAND_LINE_COUNT_SCAN_BYTES
        try:
            paths = self.tools.get("list_files", lambda workspace: [])(self.workspace)
            for path in paths if isinstance(paths, list) else []:
                full_path = os.path.join(self.workspace, path)
                try:
                    stat = os.stat(full_path)
                    normalized = str(path).replace("\\", "/")
                    signature = (stat.st_mtime_ns, stat.st_size)
                    previous = (self._terminal_file_state or {}).get(normalized)
                    lines = previous["lines"] if previous and previous["signature"] == signature else None
                    if (lines is None and stat.st_size <= MAX_COMMAND_LINE_COUNT_FILE_BYTES
                            and stat.st_size <= scan_budget):
                        with open(full_path, "r", encoding="utf-8", errors="replace") as file:
                            lines = sum(1 for _ in file)
                        scan_budget -= stat.st_size
                    snapshot[normalized] = {"signature": signature, "lines": lines}
                except OSError:
                    continue
        except Exception:
            return None
        self._terminal_file_state = snapshot
        return snapshot

    def _emit_command_file_changes(self, before):
        after = self._workspace_file_snapshot()
        if before is None or after is None:
            return
        for path in before.keys() - after.keys():
            self.emit_event("file.activity", action="deleted", path=path, lines=None,
                            removed=before[path].get("lines"))
        for path in after.keys() - before.keys():
            lines = after[path].get("lines")
            self.emit_event("file.activity", action="created", path=path, lines=lines, added=lines)
        for path in before.keys() & after.keys():
            if before[path]["signature"] != after[path]["signature"]:
                self.emit_event("file.activity", action="modified via command", path=path,
                                lines=after[path].get("lines"), added=None, removed=None)

    def _status_for_tool(self, name, arguments):
        path = arguments.get("path", "") if isinstance(arguments, dict) else ""
        query = arguments.get("query", "") if isinstance(arguments, dict) else ""
        if name == "list_files":
            return "Inspecting workspace"
        if name == "read_file":
            return f"Reading {path}"
        if name == "write_file":
            return f"Writing {path}"
        if name == "edit_file":
            return f"Editing {path}"
        if name == "search_files":
            return f"Searching {query}"
        if name == "run_command":
            return "Running command"
        if name == "create_folder":
            return f"Creating folder {path}"
        return f"Running tool {name}"

    def _tool_message(self, tool_call):
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        if not isinstance(arguments, dict):
            result = {"success": False, "error": "Arguments must be a JSON object."}
        else:
            self.show_status(self._status_for_tool(name, arguments))
            result = self.execute_tool(name, arguments)

        if isinstance(result, dict) and result.get("success") is False:
            self.show_status("Command failed; analyzing error")

        try:
            output = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            output = json.dumps({"result": str(result)}, ensure_ascii=False)

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output[:MAX_TOOL_RESULT_CHARS],
        }

    def _run_independent_reads(self, tool_calls):
        """Run independent read-only tool calls concurrently; retain model call order."""
        allowed = {"list_files", "read_file", "search_files"}
        if self.max_workers <= 1 or len(tool_calls) < 2 or any(call.function.name not in allowed for call in tool_calls):
            return {}
        worker_count = min(self.max_workers, len(tool_calls))
        results, futures = {}, {}
        started = time.monotonic()
        timed_out_workers = set()
        timeout = max(0.1, float(self.config.get("parallel_tool_timeout_seconds", 120)))
        for index, call in enumerate(tool_calls):
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            self.emit_event("worker.queued", worker_id=f"read-{index + 1}",
                            task=str(arguments.get("path", arguments.get("query", "workspace"))),
                            operation=call.function.name, status="QUEUED")
        pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="codex-read")
        pending = set()
        try:
            for index, call in enumerate(tool_calls):
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                worker_id = f"read-{index + 1}"
                operation = call.function.name
                description = str(arguments.get("path", arguments.get("query", "workspace")))
                def run_one(tool_call=call, identifier=worker_id, label=description, tool=operation):
                    began = time.monotonic()
                    self.emit_event("worker.started", worker_id=identifier, task=label,
                                    operation=tool, status="RUNNING")
                    try:
                        message = self._tool_message(tool_call)
                        tool_result = json.loads(message["content"])
                        if isinstance(tool_result, dict) and tool_result.get("success") is False:
                            status, summary = "FAILED", str(tool_result.get("error", "tool failed"))[:100]
                        else:
                            status, summary = "COMPLETED", "read complete"
                    except Exception as error:
                        message = {"role": "tool", "tool_call_id": tool_call.id,
                                   "content": json.dumps({"success": False, "error": f"{type(error).__name__}: {redact_secrets(str(error))}"})}
                        status, summary = "FAILED", type(error).__name__
                    if identifier not in timed_out_workers:
                        self.emit_event("worker.finished", worker_id=identifier, status=status,
                                        duration=time.monotonic() - began, summary=summary)
                    return message
                futures[pool.submit(run_one)] = (index, worker_id)
            done, pending = wait(futures, timeout=timeout)
            for future in done:
                index, _worker_id = futures[future]
                results[index] = future.result()
            for future in pending:
                index, worker_id = futures[future]
                timed_out_workers.add(worker_id)
                future.cancel()
                results[index] = {"role": "tool", "tool_call_id": tool_calls[index].id,
                                  "content": json.dumps({"success": False,
                                    "error": f"Read-only tool operation timed out after {timeout:g}s"})}
                self.emit_event("worker.finished", worker_id=worker_id, status="FAILED",
                                duration=timeout, summary="timed out; cancellation requested")
        finally:
            pool.shutdown(wait=not pending, cancel_futures=True)
        self.last_task_report.setdefault("performance", []).append({
            "kind": "parallel_read_batch", "workers": worker_count,
            "duration_seconds": time.monotonic() - started,
        })
        return results

    def _assistant_tool_call_message(self, tool_calls):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }

    def integrate(self, task, reports=None, coder_provider=None):
        """Run the actual coding task until the model finishes or hits the step cap."""
        self.show_status("Preparing implementation")
        provider = str(coder_provider or self.config.get("provider", "ollama" if self.config.get("execution_mode") == "offline" else "groq")).lower()
        client = self.get_client("coder", provider=provider) if coder_provider else self.get_client("coder")
        if self._active_task_deadline is None:
            self._active_task_deadline = time.monotonic() + self.task_deadline_seconds
        reports = reports if isinstance(reports, dict) else {}
        previous_report = reports.get("previous_task_report", {})
        previous_execution = self.last_task_report if isinstance(self.last_task_report, dict) else {}
        self.last_task_report = {
            "task_id": self.current_task_id,
            "task": task,
            "changed_files": [],
            "commands": [],
            "static_checks": [],
            "specification": reports.get("specification"),
            "architect": reports.get("architect"),
            "coder": {"provider": provider, "model": getattr(client, "model", "configured")},
            "escalations": list(reports.get("escalations", [])),
            "tool_events": list(self.task_tool_events),
            "llm_requests": list(previous_execution.get("llm_requests", [])),
            "llm_attempts": list(previous_execution.get("llm_attempts", [])),
            "tool_durations": list(previous_execution.get("tool_durations", [])),
        }

        tools_description = ", ".join(sorted(self.tools))
        local_json_mode = (
            provider in {"ollama", "local"}
            and bool(self.config.get("ollama_force_json_tools", True))
        )
        if provider in {"ollama", "local"}:
            # Small local models perform better with a compact, non-duplicated prompt.
            # The full tool schemas are supplied by OllamaClient's JSON protocol.
            system_prompt = f"""
You are Local Codex, an autonomous coding agent.
Workspace root: {self.workspace}
Tools available: {tools_description}

Understand the entire request in English, Hindi, or Hinglish. Finish the user's actual goal, not just the next tool action.
For a coding/build request, create complete, useful, runnable implementation code, not empty files, placeholders, TODOs, or explanations.
Inspect the existing project briefly, infer sensible defaults, and preserve its language/framework. For a simple new app with no stack, choose a practical default and implement it.
IMPORTANT: write_file automatically creates parent folders. Prefer writing the actual source file directly; do NOT spend a step creating a folder when the file write can create it.
Plan the minimum deliverables, then write complete code AND task-specific behavioral tests. Tests must exercise real behavior with representative inputs and edge cases (for a calculator, assert all four operations and division by zero; for a game, simulate player input, state updates, scoring/collisions and restart). Run the tests after implementation, inspect the actual output, fix every failure, and rerun.
For GUI applications, test core logic headlessly where possible and perform a bounded launch/smoke check when the environment supports it. Do not claim you manually played a game or interacted with a GUI unless you actually did. If GUI interaction is unavailable, clearly report the headless tests and the unverified GUI portion.
Do not run old unrelated tests and call the new task complete. Syntax checks/builds alone do NOT prove behavior. Never claim success without evidence from a test runner that discovered and passed at least one test for the current task.
Avoid repeating actions and avoid spending steps on folder/list operations once enough context is known.
Keep file operations inside the workspace. Ask only if an essential requirement cannot be inferred.
"""
        else:
            system_prompt = f"""
You are Local Codex, an autonomous software engineering agent working in:
{self.workspace}

Your job is to actually complete the user's coding task, not just explain how.
Available tools: {tools_description}

MANDATORY WORKFLOW:
1. Inspect the workspace and relevant files before changing code.
2. Make changes using the file tools; do not merely print code for the user to copy.
3. Use run_command to run relevant tests, linters, or the program when practical.
4. Read error output, diagnose the cause, edit the code, and run the check again.
5. Verify important outputs by reading the created/changed files or listing the result.
6. Continue using tools until the task is implemented and checked.
7. Never claim a file was created, a test passed, or a bug was fixed unless tool output verifies it.
8. Prefer small, focused changes. Preserve existing project conventions and unrelated work.
9. Ask a short question only when an essential requirement cannot be inferred safely.
10. Do not expose chain-of-thought. Give concise progress updates through tool use and a final summary.
11. For ordinary underspecified coding requests (for example, "make this GUI better"), infer sensible defaults, inspect the project, and implement them instead of asking the user to restate an obvious goal.
12. Treat the entire user message beginning with "Task:" as one complete task. Preserve all requirements, even if the prompt contains multiple paragraphs.
13. Before finishing, verify requested files exist and run task-specific behavioral tests, not only syntax checks. Tests must use sample inputs and edge cases; game tests should simulate key/button input and gameplay state changes. For GUIs, test logic headlessly and run a bounded GUI launch smoke test if possible. Never claim manual GUI interaction unless it occurred.
14. "Successfully done" is allowed only when the changed files pass static checks AND a relevant test runner demonstrably discovered and passed at least one test for this task. If that evidence is missing or any relevant test fails, report PARTIAL/FAILED with the exact blocker; do not present the task as successful.

TOOL CALL FORMAT:
- Prefer native function tool calls.
- If native calls are unavailable, call tools using one JSON object with keys "name" and "arguments".
- Use exact registered tool names and provide arguments as a JSON object.
- Never print a tool-call JSON object as your final answer.
- Tool results are real execution results. Use them to decide the next action.

Workspace root is the current directory. Keep file operations inside it.
"""

        messages = [{"role": "system", "content": system_prompt}]
        hosted_coder = provider not in {"ollama", "local"}
        if not hosted_coder:
            messages.extend(self.session_history[-(MAX_SESSION_TURNS * 2):])
        contract = reports.get("specification")
        contract_text = ""
        if contract:
            contract_text = (
                "\n\nIMPLEMENTATION CONTRACT (validated JSON; inspect the workspace before edits):\n"
                + redact_secrets(json.dumps(contract, ensure_ascii=False))
                + "\nInclude the acceptance criteria in your plan, create each expected file, run the specified checks, and do not claim completion without actual tool results."
            )
        repair_context = reports.get("repair_context", "")
        if repair_context:
            contract_text += "\n\nPREVIOUS LOCAL ATTEMPT AND ACTUAL VERIFICATION RESULTS:\n" + redact_secrets(str(repair_context))
        messages.append({
            "role": "user",
            "content": (
                f"Task: {redact_secrets(task)}{contract_text}\n\n"
                "Work autonomously. Inspect, implement, run checks, fix failures, and verify."
            ),
        })
        base_system_prompt = system_prompt
        self._context_dirty = True

        def refresh_system_context():
            if not self._context_dirty:
                return
            current_context = self._workspace_context_text(
                task,
                previous_report=previous_report,
                specification=contract,
            )
            messages[0] = {
                "role": "system",
                "content": base_system_prompt + "\n\nCURRENT WORKSPACE CONTEXT:\n" + current_context,
            }
            self._context_dirty = False

        malformed_tool_retries = 0
        completion_retries = 0
        empty_response_retries = 0
        disable_tools_next = False
        successful_mutation_calls = set()
        failed_command_calls = {}
        mutation_generation = 0
        executed_tool_calls = 0
        mutation_needs_verification = False
        last_action_signature = None
        repeated_action_count = 0
        local_fallback_attempted = False
        recent_user_tasks = [
            item.get("content", "")
            for item in self.session_history
            if item.get("role") == "user"
        ][-2:]
        # Deliverable tracking must describe THIS task only. Including previous chat
        # requests here caused stale files/tests to be treated as current requirements.
        task_context = task
        consecutive_no_progress = 0
        no_progress_recoveries = 0

        deadline_reached = False
        for _ in range(self.max_steps):
            if time.monotonic() >= self._active_task_deadline:
                deadline_reached = True
                self.show_status("Task deadline reached; stopping and reporting verified work.")
                break
            self.show_status("Working")
            self.emit_event("step.started", phase="model")
            refresh_system_context()
            request_tools = None if disable_tools_next else self.tool_schemas
            disable_tools_next = False
            response = self._chat_with_events(
                client, messages, tools=request_tools, provider=provider,
            )
            usage = self._usage_metadata(response)
            if usage:
                self.last_task_report.setdefault("usage", []).append({
                    "provider": provider,
                    "model": getattr(client, "model", "configured"),
                    **usage,
                })
            message = response.choices[0].message
            native_calls = getattr(message, "tool_calls", None) or []
            tool_calls = list(native_calls)
            recovered = False

            if not tool_calls:
                tool_calls = recover_text_tool_calls(message.content, self.tools)
                recovered = bool(tool_calls)

            if not tool_calls:
                raw_content = getattr(message, "content", None)
                final_text = (raw_content or "").strip()

                # Small local models occasionally return an empty assistant message
                # when tool schemas are included. Retry with a shorter instruction and
                # no schemas once, allowing the model to emit a recoverable JSON call.
                if not final_text and empty_response_retries < 3:
                    empty_response_retries += 1
                    if raw_content is not None:
                        messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. Continue the coding task. "
                            "Do not return an empty response. Choose the next action and "
                            "use exactly one JSON object like "
                            '{"name":"list_files","arguments":{}} '
                            "with a registered tool name and an arguments object, or provide "
                            "a concise final answer only after the task is complete."
                        ),
                    })
                    disable_tools_next = True
                    continue

                if not final_text:
                    final_text = (
                        "The local model returned an empty response after 3 retries. "
                        "Check that Ollama is running and the selected model is installed "
                        "with 'ollama list'. Very small models may struggle with multi-step "
                        "coding; try a stronger local coding model if memory permits."
                    )

                if (
                    _is_tool_call_shaped_json(final_text)
                    and malformed_tool_retries < 2
                ):
                    malformed_tool_retries += 1
                    messages.append({"role": "assistant", "content": final_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "That looks like a tool call but was not executable. "
                            f"Use only these exact registered tools: {tools_description}. "
                            "Return a native function call or JSON with keys name and arguments. "
                            "Arguments must be an object. Continue the task; do not stop."
                        ),
                    })
                    continue

                missing_artifacts = []
                expected_paths = self._task_expected_paths(task_context)
                for artifact in expected_paths:
                    full_path = os.path.realpath(os.path.join(self.workspace, artifact))
                    workspace_root = os.path.realpath(self.workspace)
                    try:
                        inside_workspace = (
                            os.path.commonpath([workspace_root, full_path])
                            == workspace_root
                        )
                    except ValueError:
                        inside_workspace = False
                    if inside_workspace and not os.path.exists(full_path):
                        missing_artifacts.append(artifact)

                echoed_tool_result = _is_tool_result_json(final_text)
                no_local_work = local_json_mode and executed_tool_calls == 0
                unverified_changes = local_json_mode and mutation_needs_verification
                spec_issues = self._task_completeness_issues() if self.last_task_report.get("specification") else []
                # A local model gets one chance to continue naturally. If it still
                # claims completion with missing files, switch to constrained recovery.
                completion_retry_limit = 1 if local_json_mode else 4
                if (
                    (echoed_tool_result or missing_artifacts or no_local_work or unverified_changes or spec_issues)
                    and completion_retries < completion_retry_limit
                ):
                    completion_retries += 1
                    messages.append({"role": "assistant", "content": final_text})

                    reasons = []
                    if echoed_tool_result:
                        reasons.append(
                            "Your last message is a tool-result JSON object, not a user-facing final answer."
                        )
                    if missing_artifacts:
                        reasons.append(
                            "Required deliverables still missing: "
                            + ", ".join(missing_artifacts)
                            + "."
                        )
                    if no_local_work:
                        reasons.append(
                            "You have not executed any workspace tool yet. Inspect the project and do the requested work."
                        )
                    if unverified_changes:
                        reasons.append(
                            "You changed a source file but have not verified it with a relevant run_command check. Run the check, fix failures, and rerun it."
                        )
                    if spec_issues:
                        reasons.append("Acceptance criteria are not yet satisfied: " + "; ".join(spec_issues) + ".")

                    messages.append({
                        "role": "user",
                        "content": (
                            "The task is NOT complete. "
                            + " ".join(reasons)
                            + " Continue working now using the registered tools. "
                            "Create any missing files/folders, put the actual implementation in the files, "
                            "then verify them with read_file or run_command. Do not merely repeat tool output "
                            "or say 'Task completed'. Once everything exists and is verified, give a concise summary."
                        ),
                    })
                    continue

                if echoed_tool_result or missing_artifacts or no_local_work or unverified_changes or spec_issues:
                    details = []
                    if echoed_tool_result:
                        details.append("the model repeatedly echoed tool-result JSON")
                    if missing_artifacts:
                        details.append(
                            "missing deliverables: " + ", ".join(missing_artifacts)
                        )
                    if no_local_work:
                        details.append("no workspace tools were executed")
                    if unverified_changes:
                        details.append("source changes were not verified with a successful command")
                    if spec_issues:
                        details.append("acceptance criteria failed: " + "; ".join(spec_issues))
                    final_text = (
                        "Local Codex could not verify task completion because "
                        + "; ".join(details)
                        + ". The model may be too small for this task. "
                        "Try Groq or a stronger local coding model, then ask it to continue."
                    )

                if local_json_mode and (missing_artifacts or spec_issues) and not local_fallback_attempted:
                    local_fallback_attempted = True
                    self.show_status("Model stopped before implementation; generating missing files")
                    recovery_paths = list(expected_paths)
                    recovered_files = self._generate_missing_files_locally(
                        client, task_context, recovery_paths
                    )
                    if recovered_files:
                        summary = "Generated missing implementation files after the model stopped prematurely, then ran verification."
                        self.session_history.extend([
                            {"role": "user", "content": redact_secrets(task)},
                            {"role": "assistant", "content": summary},
                        ])
                        self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                        return self._format_final_report(summary)
                    final_text = (
                        "The model stopped before completing the requested files. "
                        "Automatic file-generation recovery was attempted but did not pass verification. "
                        "The task is NOT marked successful. "
                        + final_text
                    )

                self.session_history.extend([
                    {"role": "user", "content": redact_secrets(task)},
                    {"role": "assistant", "content": final_text},
                ])
                self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                return self._format_final_report(final_text)

            # In local JSON-tool mode, keep the transcript as ordinary user/assistant
            # messages. Synthetic assistant tool_calls + role=tool messages are often
            # mishandled by small Ollama models because no native tool schema was sent.
            if local_json_mode and recovered:
                messages.append({
                    "role": "assistant",
                    "content": getattr(message, "content", None) or "",
                })
            elif recovered:
                messages.append(self._assistant_tool_call_message(tool_calls))
            else:
                messages.append(message)

            # Track real implementation progress separately from tool-call activity.
            # Creating folders, listing files, or running unrelated commands is not progress.
            productive_step = False
            # Execute one-action-at-a-time tool requests and give local models readable
            # results instead of a synthetic tool-call protocol they may echo forever.
            blocked_duplicate_action = False
            parallel_reads = self._run_independent_reads(tool_calls)
            for tool_index, tool_call in enumerate(tool_calls):
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                signature = (
                    name,
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                )
                if signature == last_action_signature:
                    repeated_action_count += 1
                else:
                    last_action_signature = signature
                    repeated_action_count = 1

                duplicate_successful_mutation = (
                    name in {"create_folder", "write_file", "edit_file"}
                    and signature in successful_mutation_calls
                )
                duplicate_failed_command = (
                    name == "run_command"
                    and failed_command_calls.get(signature) == mutation_generation
                )
                if duplicate_successful_mutation or duplicate_failed_command:
                    tool_result = {
                        "success": False,
                        "error": (
                            "Repeated action blocked: this command already failed and no source file "
                            "has changed since. Fix the implementation or choose a different check."
                            if duplicate_failed_command else
                            "Duplicate mutation blocked: this exact action already succeeded. Choose the next missing step."
                        ),
                    }
                    blocked_duplicate_action = True
                    tool_message = None
                    if duplicate_failed_command:
                        self.emit_event(
                            "activity",
                            message="Repeated action detected; trying a different check or fixing the source first.",
                        )
                else:
                    tool_message = parallel_reads.get(tool_index) or self._tool_message(tool_call)
                    try:
                        tool_result = json.loads(tool_message["content"])
                    except (json.JSONDecodeError, TypeError):
                        tool_result = {"success": False, "error": tool_message["content"]}

                    if (
                        name in {"create_folder", "write_file", "edit_file"}
                        and isinstance(tool_result, dict)
                        and tool_result.get("success") is True
                    ):
                        successful_mutation_calls.add(signature)
                    if name in {"write_file", "edit_file"} and isinstance(tool_result, dict) and tool_result.get("success") is True:
                        mutation_generation += 1
                    if name == "run_command" and isinstance(tool_result, dict):
                        if tool_result.get("success") is True and tool_result.get("return_code") == 0:
                            failed_command_calls.pop(signature, None)
                        elif tool_result.get("success") is False or tool_result.get("return_code") not in {None, 0}:
                            failed_command_calls[signature] = mutation_generation

                executed_tool_calls += 1
                if (
                    name in {"write_file", "edit_file"}
                    and isinstance(tool_result, dict)
                    and tool_result.get("success") is True
                ):
                    productive_step = True
                if (
                    name in {"write_file", "edit_file"}
                    and isinstance(tool_result, dict)
                    and tool_result.get("success") is True
                ):
                    path = arguments.get("path", "")
                    if path:
                        if path not in self.last_task_report["changed_files"]:
                            self.last_task_report["changed_files"].append(path)
                        verification = verify_changed_file(self.workspace, path)
                        self.last_task_report["static_checks"].append(verification)
                        self.emit_event("verification", message=f"Static check {path}: {'PASS' if verification.get('success') else 'FAIL'}")
                        tool_result["verification"] = verification
                        mutation_needs_verification = True
                        if not verification.get("success"):
                            self.show_status(f"Automatic verification failed for {path}")
                if name == "run_command" and isinstance(tool_result, dict) and not blocked_duplicate_action:
                    command = redact_secrets(str(arguments.get("command", "")))
                    record = {
                        "command": command,
                        "cwd": tool_result.get("cwd", self.workspace),
                        "duration_seconds": tool_result.get("duration_seconds"),
                        "timed_out": tool_result.get("timed_out", False),
                        "return_code": tool_result.get("return_code"),
                        "success": tool_result.get("success") is True,
                        "stdout": tool_result.get("stdout", ""),
                        "stderr": tool_result.get("stderr", ""),
                        "meaningful_check": _is_meaningful_verification_command(command),
                    }
                    record["test_counts"] = _test_counts(
                        command, record["stdout"], record["stderr"]
                    )
                    self.last_task_report["commands"].append(record)
                    record["executed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    self.workspace_memory.record_command(record)
                    if record["meaningful_check"]:
                        self.emit_event("verification", message=f"{command} — {'PASS' if record['success'] and record['return_code'] == 0 else 'FAIL'}")
                    if (
                        record["meaningful_check"]
                        and record["success"]
                        and record["return_code"] == 0
                    ):
                        mutation_needs_verification = False
                        productive_step = True

                if local_json_mode and recovered:
                    readable_result = json.dumps(
                        tool_result, ensure_ascii=False, default=str
                    )[:MAX_TOOL_RESULT_CHARS]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"RESULT of tool {name} with arguments "
                            f"{json.dumps(arguments, ensure_ascii=False)}: "
                            f"{readable_result}\n\n"
                            "Continue the original task from this real result. "
                            "Choose the next distinct action as exactly one JSON object. "
                            "Do not repeat a successful action. If implementation is written, "
                            "run a relevant check and inspect the result before finishing."
                        ),
                    })
                elif blocked_duplicate_action:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                else:
                    messages.append(tool_message)

            if blocked_duplicate_action:
                messages.append({
                    "role": "user",
                    "content": (
                        "A repeated action was blocked. Continue the original task with a distinct "
                        "action; do not repeat the same failing command without changing source code. "
                        "Write the actual implementation, then run a syntax check/test and "
                        "inspect the result. Finish only after verifying the workspace."
                    ),
                })

            # Recover not only from identical repeated calls but also from agents
            # that keep doing non-productive actions (folders/listing/unrelated commands).
            # This prevents wasting the entire configured step budget.
            consecutive_no_progress = 0 if productive_step else consecutive_no_progress + 1
            should_recover = repeated_action_count >= 3 or consecutive_no_progress >= 4
            if (
                local_json_mode
                and not local_fallback_attempted
                and should_recover
            ):
                missing = self._missing_expected_artifacts(task_context)
                if missing:
                    local_fallback_attempted = True
                    self.show_status("Repeated action detected; recovering missing deliverables")
                    recovered_files = self._generate_missing_files_locally(
                        client,
                        task_context,
                        self._task_expected_paths(task_context),
                    )
                    if recovered_files:
                        summary = (
                            "Recovered from a repeated tool-call loop. Requested files were written "
                            "and the available verification command was run."
                        )
                        self.session_history.extend([
                            {"role": "user", "content": redact_secrets(task)},
                            {"role": "assistant", "content": summary},
                        ])
                        self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                        return self._format_final_report(summary)
                    return self._format_final_report(
                        "Local Codex detected a repeated tool-call loop and tried file-generation recovery, "
                        "but it could not produce all requested files with a passing verification command. "
                        "The task is NOT marked successful; inspect the reported checks and try a stronger model."
                    )
            elif should_recover:
                if no_progress_recoveries >= 1:
                    return self._format_final_report(
                        "The model repeated actions without making verifiable progress after a recovery prompt. "
                        "The task is NOT marked successful; inspect the reported missing files and checks."
                    )
                no_progress_recoveries += 1
                repeated_action_count = 0
                consecutive_no_progress = 0
                self.emit_event(
                    "activity",
                    message="Repeated or unproductive action detected; asking the selected model for one different, verifiable action.",
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your recent actions have not made verifiable progress. Do not repeat them. "
                        "Use the actual tool results, make one concrete change to a required deliverable, "
                        "then run a relevant verification command. If you cannot continue, explain the blocker."
                    ),
                })

        if deadline_reached:
            self.last_task_report["deadline_reached"] = True
            self.last_task_report["deadline_seconds"] = self.task_deadline_seconds
            self.last_task_report["verification_status"] = (
                "PARTIAL" if self.last_task_report.get("changed_files") else "BLOCKED"
            )
            return self._format_final_report(
                f"Task deadline reached after {self.task_deadline_seconds} seconds. "
                "The task is not marked complete unless its recorded checks passed."
            )

        # The step cap is a safety net, not an excuse to leave a coding task half done.
        # Before reporting a limit, make one constrained recovery attempt for missing files.
        missing_at_limit = self._missing_expected_artifacts(task_context)
        if local_json_mode and not local_fallback_attempted and missing_at_limit:
            self.show_status("Step budget reached; recovering missing implementation files")
            local_fallback_attempted = True
            recovered_files = self._generate_missing_files_locally(
                client, task_context, self._task_expected_paths(task_context)
            )
            if recovered_files:
                summary = "Recovered the missing implementation after the normal tool loop made insufficient progress."
                self.session_history.extend([
                    {"role": "user", "content": redact_secrets(task)},
                    {"role": "assistant", "content": summary},
                ])
                self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                return self._format_final_report(summary)

        limit_message = (
            f"Stopped after {self.max_steps} tool steps. "
            "The recovery attempt could not verify all requested deliverables; the task is not marked successful."
        )
        self.session_history.extend([
                    {"role": "user", "content": redact_secrets(task)},
            {"role": "assistant", "content": limit_message},
        ])
        self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
        return self._format_final_report(limit_message)

    def explain(self, question):
        """Answer a code question from retrieved workspace evidence without tools."""
        related = references_existing_context(question)
        previous = self.last_task_report if related else {}
        self.workspace_memory.refresh()
        context = self.workspace_context.build_context(
            task=question,
            previous_report=previous,
            recent_history=self.session_history if related else [],
            recent_tool_calls=self.recent_tool_calls if related else [],
            specification=None,
            refresh=False,
        )
        context += "\n\n" + self.workspace_memory.context_for(
            question, max_chars=int(self.config.get("workspace_memory_context_chars", 4200)), refresh=False
        )
        provider = str(self.config.get("provider", "ollama" if self.config.get("execution_mode") == "offline" else "groq")).lower()
        client = self.get_client("coder")
        self.emit_event("task.started", task=question, workspace=self.workspace)
        self.emit_event("workspace.memory", **self.workspace_memory.dashboard_state())
        try:
            response = self._chat_with_events(
                client,
                [
                    {"role": "system", "content": (
                        "You are Local Codex answering a read-only question about the current codebase. "
                        "Use only the supplied inspected workspace evidence. Do not call tools, suggest an unverified cause as fact, "
                        "or claim tests ran. Say which evidence is missing when the answer cannot be verified."
                    )},
                    {"role": "user", "content": (
                        f"Question: {redact_secrets(question)}\n\nINSPECTED WORKSPACE CONTEXT:\n{context}"
                    )},
                ],
                tools=None, provider=provider, phase="explanation",
            )
            answer = self._response_text(response) or "I could not produce an answer from the available workspace evidence."
            self.session_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
            return answer
        finally:
            self.emit_event("task.finished", status="READ ONLY")

    def run(self, task):
        """Architect the request, run the local agent, then escalate once if configured."""
        self.current_task_id = uuid.uuid4().hex
        task_started = time.monotonic()
        self._active_task_deadline = task_started + self.task_deadline_seconds
        self.emit_event("task.started", task=task, workspace=self.workspace)
        self.show_status("Inspecting workspace and loading project context")
        self.workspace_memory.refresh()
        self.workspace_memory.record_task_start(task, self.current_task_id)
        request_kind = classify_request(
            task, has_history=bool(self.session_history),
            has_project=bool(self.workspace_context.files()),
        )
        prior_report = dict(self.last_task_report) if self.last_task_report else {}
        self.last_task_report = {
            "task_id": self.current_task_id,
            "task": task,
            "llm_requests": [],
            "tool_durations": [],
        }
        is_follow_up = bool(prior_report or self.session_history) and (
            request_kind == "follow_up"
            or references_existing_context(task)
            or request_kind == "bug_report" and bool(self.session_history)
        )
        if not is_follow_up:
            self.session_history.clear()
            self.recent_tool_calls.clear()
            prior_report = {}
        self.task_tool_events.clear()
        self.current_plan = []
        provider = str(self.config.get("provider", "ollama" if self.config.get("execution_mode") == "offline" else "groq")).lower()
        if provider in {"nvidia", "nvidia_nim"}:
            model = self.config.get("nvidia_model", "qwen/qwen2.5-coder-32b-instruct")
        elif provider in {"ollama", "local"}:
            model = self.config.get("local_model", "qwen2.5-coder:1.5b")
        else:
            model = self.config.get("models", {}).get("coder", "not configured")
        try:
            # Common calculator requests are deterministic and need no LLM planning.
            # This avoids repeated low-information tool calls from tiny local models.
            if supports_calculator_project(task):
                result = self._recover_calculator_gui_task(task)
                return result

            self.show_status("Planning task and identifying deliverables")
            specification, architect = self._architect_task(task, previous_report=prior_report)
            specification["original_request"] = task
            specification["workspace_path"] = self.workspace
            specification["expected_directory"] = next(
                (item["path"].split("/", 1)[0] for item in specification.get("expected_files", [])
                 if isinstance(item, dict) and "/" in item.get("path", "")), None,
            )
            self.current_plan = list(specification.get("acceptance_criteria", [])) + [
                item.get("path") for item in specification.get("expected_files", []) if isinstance(item, dict)
            ]
            self.workspace_memory.record_task_start(task, self.current_task_id, specification)
            orchestration = {
                "specification": specification, "architect": architect, "escalations": [],
                "previous_task_report": prior_report,
            }
            try:
                self.show_status("Implementing task")
                result = self.integrate(task, orchestration)

                # Some provider clients convert timeouts into a normal failed
                # task report instead of raising. Recover supported car-game
                # requests on that path too; previously the recovery ran only
                # when integrate() raised, leaving the task unimplemented.
                completed_report = self.last_task_report if isinstance(self.last_task_report, dict) else {}
                failed_requests = [
                    item for item in completed_report.get("llm_requests", [])
                    if isinstance(item, dict) and item.get("success") is False
                ]
                architect_status = str(completed_report.get("architect", {}).get("status", "")).upper()
                coder_status = str(completed_report.get("coder", {}).get("status", "")).upper()
                failure_evidence = bool(
                    failed_requests
                    or completed_report.get("initial_error")
                    or architect_status in {"FAILED", "FAIL", "ERROR"}
                    or coder_status in {"FAILED", "FAIL", "ERROR"}
                )
                if completed_report.get("verification_status") != "PASS" and (
                    failure_evidence or supports_car_game_recovery(task)
                ):
                    # For the explicitly supported car-racing task, failed gameplay
                    # acceptance checks are enough to trigger the deterministic
                    # implementation; waiting for an HTTP/provider error left
                    # placeholder games untouched.
                    failure = (
                        (failed_requests[-1].get("error") if failed_requests else None)
                        or completed_report.get("initial_error")
                        or (
                            "Game acceptance checks failed: "
                            + "; ".join(completed_report.get("completeness_issues", []))
                        )
                        or f"Provider task failed (architect={architect_status or 'unknown'}, coder={coder_status or 'unknown'})."
                    )
                    recovered_result = self._recover_car_game_after_provider_failure(
                        task, specification, failure
                    )
                    if recovered_result is not None:
                        result = recovered_result
            except Exception as error:
                previous_execution = self.last_task_report if isinstance(self.last_task_report, dict) else {}
                self.last_task_report = {
                    **previous_execution,
                    "task": task,
                    "changed_files": list(previous_execution.get("changed_files", [])),
                    "commands": list(previous_execution.get("commands", [])),
                    "static_checks": list(previous_execution.get("static_checks", [])),
                    "specification": specification,
                    "architect": architect,
                    "coder": previous_execution.get(
                        "coder", {"provider": provider, "model": model}
                    ),
                    "escalations": list(previous_execution.get("escalations", [])),
                    "initial_error": f"{type(error).__name__}: {redact_secrets(str(error))}",
                    "verification_status": previous_execution.get(
                        "verification_status",
                        "PARTIAL" if previous_execution.get("changed_files") else "FAIL",
                    ),
                }
                recovered_result = self._recover_car_game_after_provider_failure(
                    task, specification, error
                )
                result = recovered_result or self._format_final_report(
                    "The selected coding provider failed before implementation. No verified implementation is available yet, and this task is NOT marked complete. Check the provider API key/model or restart and select another of the four configured coding models."
                )

            max_escalations = max(0, min(int(self.config.get("max_escalations", 1)), 2))
            escalation_provider = self._escalation_provider(provider)
            if (
                self.last_task_report.get("verification_status") != "PASS"
                and max_escalations > 0
                and self.config.get("hosted_coder_escalation", True)
                and escalation_provider
            ):
                previous = self.last_task_report
                context = {
                    "task": task,
                    "specification": specification,
                    "files": {},
                    "commands": previous.get("commands", [])[-6:],
                    "static_checks": previous.get("static_checks", [])[-12:],
                    "completeness_issues": previous.get("completeness_issues", []),
                    "initial_error": previous.get("initial_error"),
                }
                read_tool = self.tools.get("read_file")
                if read_tool:
                    for path in previous.get("changed_files", [])[:8]:
                        try:
                            content = read_tool(self.workspace, path)
                            context["files"][path] = redact_secrets(str(content)[:12000])
                        except Exception as error:
                            context["files"][path] = f"Unable to read changed file ({type(error).__name__})."
                repair_context = json.dumps(context, ensure_ascii=False, default=str)[:40000]
                self.show_status("Repairing implementation after verification failure")
                escalation = {"provider": escalation_provider, "status": "STARTED"}
                retry_context = dict(orchestration)
                retry_context["repair_context"] = repair_context
                retry_context["escalations"] = [escalation]
                try:
                    result = self.integrate(task, retry_context, coder_provider=escalation_provider)
                except Exception as error:
                    self.last_task_report = {
                        "task": task, "changed_files": previous.get("changed_files", []),
                        "commands": previous.get("commands", []), "static_checks": previous.get("static_checks", []),
                        "specification": specification, "architect": architect,
                        "coder": {"provider": escalation_provider, "model": self._model_for(escalation_provider, "coder")},
                        "escalations": [{"provider": escalation_provider, "status": "FAILED", "error": redact_secrets(str(error))[:1200]}],
                        "initial_error": redact_secrets(str(error))[:1200],
                        "verification_status": "PARTIAL",
                    }
                    result = self._format_final_report("Hosted coder escalation failed; the task remains unverified.")
        finally:
            if not self.last_task_report:
                self.last_task_report = {
                    "task_id": self.current_task_id, "task": task, "changed_files": [],
                    "commands": [], "static_checks": [], "verification_status": "FAIL",
                    "initial_error": "Task ended before an execution report was produced.",
                }
            if self.last_task_report:
                self.last_task_report.setdefault("task_id", self.current_task_id)
                self.last_task_report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.last_task_report["duration_seconds"] = time.monotonic() - task_started
                self.last_task_report["tool_events"] = list(self.task_tool_events)
                self.last_task_report["final_report"] = redact_secrets(str(locals().get("result", "")))[:8000]
                self.last_task_report["source_changes"] = self._trajectory_source_changes(self.last_task_report)
                self.workspace_memory.refresh()
                self.workspace_memory.record_task_finish(self.last_task_report)
            self.emit_event("workspace.memory", **self.workspace_memory.dashboard_state())
            self.emit_event("task.report", report={
                "verification_status": self.last_task_report.get("verification_status", "NOT VERIFIED"),
                "commands": self.last_task_report.get("commands", []),
                "static_checks": self.last_task_report.get("static_checks", []),
                "completeness_issues": self.last_task_report.get("completeness_issues", []),
                "initial_error": self.last_task_report.get("initial_error"),
                "changed_files": self.last_task_report.get("changed_files", []),
            })
            self.show_status("Done")
            self.emit_event("task.finished", status=self.last_task_report.get("verification_status", "UNVERIFIED"))
        return result
