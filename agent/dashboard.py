"""Event-driven presentation dashboard with Rich and plain-text renderers."""

from collections import OrderedDict
from datetime import datetime, timezone
from agent.orchestration import redact_secrets
import sys
import os
import threading
import time

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # Rich is optional for source checkouts without dependencies installed.
    Console = Group = Live = Panel = Table = Text = None


class EventBus:
    def __init__(self):
        self._subscribers = []

    def subscribe(self, callback):
        self._subscribers.append(callback)
        return callback

    def emit(self, event, **data):
        payload = {"event": event, "timestamp": datetime.now(timezone.utc), **data}
        for callback in tuple(self._subscribers):
            try:
                callback(payload)
            except Exception:
                # A display plugin must never interrupt agent work.
                continue


class TerminalDashboard:
    WORKFLOW = (
        ("understanding", "Understand request"),
        ("inspecting", "Inspect workspace"),
        ("planning", "Plan"),
        ("implementing", "Implement"),
        ("testing", "Test"),
        ("repairing", "Repair errors"),
        ("verifying", "Verify result"),
    )

    def __init__(self, mode="auto", workspace="", provider="", model="", execution_mode="offline",
                 stream=None, demo_mode=True, language="english", show_technical=False):
        mode = str(mode or "auto").lower()
        if mode not in {"auto", "rich", "plain"}:
            mode = "auto"
        self.stream = stream or sys.stdout
        self.rich_enabled = bool(Live and mode != "plain" and getattr(self.stream, "isatty", lambda: False)())
        self.mode = mode
        self.demo_mode = bool(demo_mode)
        self.language = "hinglish" if str(language).lower() in {"hinglish", "hi", "hindi"} else "english"
        self.show_technical = bool(show_technical)
        self.workspace, self.provider, self.model = workspace, provider, model
        self.execution_mode = str(execution_mode).upper()
        self.bus = EventBus()
        self.bus.subscribe(self._on_event)
        self.lock = threading.RLock()
        self.activity = "Ready"
        self.current_reason = ""
        self.phase = "Ready"
        self.task_status = "Ready"
        self.explanation = "Ready for a coding task."
        self.files = OrderedDict()
        self.tools = []
        self.activity_log = []
        self.technical_log = []
        self.workers = OrderedDict()
        self.verification = []
        self.commands = []
        self.unresolved_issues = []
        self.last_error = ""
        self.created_folders = set()
        self.gui_status = "Not recorded"
        self.verification_status = "Not verified"
        self.workflow = {key: "pending" for key, _label in self.WORKFLOW}
        self._last_printed_activity = None
        self.live = None
        self.started_at = None
        self.task = ""
        self.current_file = ""
        self.memory = {"status": "Not indexed", "indexed_files": 0, "context_files": 0,
                       "unresolved_issues": 0, "last_task": "none", "stale": False,
                       "project_type": "unknown", "task_context_status": "unknown"}

    def status(self, message):
        self.bus.emit("activity", message=str(message))

    def _on_event(self, event):
        with self.lock:
            kind = event["event"]
            self._technical_event(event)
            if kind == "task.started":
                self._start_task(event)
            elif kind == "activity":
                self._handle_status(str(event.get("message", "Working")))
            elif kind == "step.started":
                # Legacy internal step events are intentionally omitted from the presentation.
                pass
            elif kind == "model.request":
                phase = event.get("phase", "model")
                if phase == "planning":
                    self._set_phase("planning", "Planning", "The agent is turning your request into a list of deliverables.")
                    message = self._say("The AI is planning the task.", "AI task ka plan bana raha hai.")
                elif phase == "explanation":
                    message = self._say("The AI is preparing an answer from the project files.", "AI project files dekh kar jawab taiyar kar raha hai.")
                else:
                    message = self._say("The AI is choosing its next coding action.", "AI agla coding action chun raha hai.")
                self.task_status = "In progress"
                self._set_activity(message, reason=f"Model: {event.get('provider', 'configured')} / {event.get('model', 'configured')}")
            elif kind == "model.response":
                ok = event.get("success", True)
                error = str(event.get("error", "")).strip()
                if error:
                    self.last_error = redact_secrets(error)
                self._set_activity(
                    "AI returned its next action." if ok else "AI request failed; the agent will handle the error.",
                    reason=(f"Model response in {self._duration(event.get('duration'))}."
                            if ok else error or f"Request failed after {self._duration(event.get('duration'))}."),
                    technical=f"Model response success={ok}" + (f"; {error}" if error else ""),
                )
            elif kind == "provider.retry":
                self.last_error = redact_secrets(str(event.get("error", "Temporary request error")))
                self._set_activity(
                    f"{event.get('provider', 'AI provider')} is temporarily unavailable; retrying.",
                    reason=(f"Retry {event.get('attempt', '?')} of {event.get('attempts', '?')} "
                            f"after {self._duration(event.get('delay'))}. "
                            f"{event.get('error', 'Temporary request error')}"),
                    technical=f"Provider retry event: {event}",
                )
            elif kind == "tool.started":
                self._handle_tool_started(event)
            elif kind == "tool.finished":
                self._handle_tool_finished(event)
            elif kind == "tool.result":
                self._handle_tool_result(event)
            elif kind in {"worker.queued", "worker.started", "worker.finished", "worker.blocked"}:
                self._handle_worker(event)
            elif kind == "file.activity":
                self._handle_file(event)
            elif kind == "verification":
                self._handle_verification(event)
            elif kind == "task.report":
                self._handle_report(event)
            elif kind == "workflow.phase":
                self._set_phase(event.get("phase"), event.get("state", "active"), event.get("message", ""))
            elif kind == "workspace.memory":
                self.memory.update({key: event[key] for key in self.memory if key in event})
            elif kind == "task.finished":
                self.verification_status = self._verification_label(event.get("status", "UNVERIFIED"))
                self.task_status = "Finished" if event.get("status") in {"PASS", "READ ONLY"} else "Finished with issues"
                if event.get("status") == "READ ONLY":
                    for key, _label in self.WORKFLOW:
                        self.workflow[key] = "complete" if key == "understanding" else "skipped"
                    self._set_phase("verifying", "skipped")
                else:
                    final_state = "complete" if event.get("status") == "PASS" else "failed"
                    for key, _label in self.WORKFLOW:
                        if self.workflow[key] == "active":
                            self.workflow[key] = final_state
                    self._set_phase("verifying", final_state)
                self._set_activity(f"Task finished: {self.verification_status}")
                self._refresh()
                if self.live:
                    self.live.stop()
                    self.live = None
                else:
                    print(self._plain_summary(), file=self.stream, flush=True)
            self._refresh()

    def _start_task(self, event):
        self.task = str(event.get("task", ""))
        self.workspace = str(event.get("workspace", self.workspace))
        self.started_at = time.monotonic()
        self.phase = "Understanding request"
        self.task_status = "In progress"
        self.explanation = "The agent is identifying what you asked it to build or change."
        self.workflow = {key: "pending" for key, _label in self.WORKFLOW}
        self.workflow["understanding"] = "active"
        self.files.clear()
        self.tools.clear()
        self.activity_log.clear()
        self.technical_log.clear()
        self.workers.clear()
        self.verification.clear()
        self.commands.clear()
        self.unresolved_issues.clear()
        self.last_error = ""
        self.created_folders.clear()
        self.current_file = ""
        self.gui_status = "Not recorded"
        self.verification_status = "Not verified"
        self._set_activity("Understanding your request", reason="The agent begins by identifying the requested outcome.")
        if self.rich_enabled and self.live is None:
            self.live = Live(self._render(), console=Console(file=self.stream),
                             refresh_per_second=4, screen=False, transient=False, auto_refresh=True)
            self.live.start()
        elif not self.rich_enabled:
            project = os.path.basename(os.path.normpath(self.workspace)) or self.workspace
            print(f"\nLOCAL CODEX | {self.provider} / {self.model} | {self.execution_mode}\nWorkspace: {project} ({self.workspace})", file=self.stream, flush=True)

    def _handle_status(self, raw):
        text = raw.strip()
        lowered = text.lower()
        if "inspect" in lowered or "workspace" in lowered and "context" in lowered:
            self._set_phase("inspecting", "Inspecting workspace", "The agent is checking project files and existing conventions.")
        elif "plan" in lowered or "deliverable" in lowered:
            self._set_phase("planning", "Planning", "The agent is deciding what needs to change.")
        elif any(word in lowered for word in ("implement", "preparing implementation", "writing", "editing", "creating")):
            self._set_phase("implementing", "Implementing", "The agent is creating or updating project files.")
        elif any(word in lowered for word in ("repair", "recover", "repeated", "failed", "stalled", "error")):
            self._set_phase("repairing", "Repairing", "The agent encountered a problem and is trying another action.")
            text = self._friendly_recovery(text)
        elif text.lower() == "done":
            self._set_phase("verifying", "Verifying result", "The agent is preparing its final result and verification status.")
        elif "test" in lowered or "verif" in lowered or "check" in lowered:
            self._set_phase("testing", "Testing", "The agent is running or reviewing a check.")
        else:
            text = self._friendly_status(text)
        self._set_activity(text, reason=self._reason_for_status(text))

    def _handle_tool_started(self, event):
        tool = str(event.get("tool", "tool"))
        message, reason = self._tool_description(tool, event.get("arguments", {}), event.get("label", ""))
        if tool in {"run_command", "verify"}:
            self._set_phase("testing", "Testing", "The agent is running a project command to check its work.")
        elif tool in {"write_file", "edit_file", "create_folder"}:
            self._set_phase("implementing", "Implementing", "The agent is changing project files.")
        self._set_activity(message, reason=reason, technical=f"Tool started: {tool}")

    def _handle_tool_finished(self, event):
        tool = str(event.get("tool", "tool"))
        success = bool(event.get("success", False))
        duration = self._duration(event.get("duration"))
        self.tools.append({"tool": tool, "duration": event.get("duration"), "success": success})
        self.tools = self.tools[-20:]
        if tool == "run_command":
            self.commands.append({"command": event.get("command", "Command"), "status": "completed" if success else "failed",
                                  "duration": event.get("duration"), "return_code": event.get("return_code")})
        message = f"{self._tool_friendly_name(tool)} {'completed' if success else 'failed'} ({duration})."
        self._set_activity(message, reason=f"Tool: {tool}; result: {'success' if success else 'failure'}.",
                           technical=f"Tool finished: {tool}, success={success}, duration={duration}")

    def _handle_tool_result(self, event):
        result = event.get("result")
        tool = str(event.get("tool", "tool"))
        if not isinstance(result, dict):
            return
        if result.get("success") is False:
            error = str(result.get("error", "Unknown error"))[:240]
            self.last_error = error
            self._set_phase("repairing", "Repairing", "A tool operation failed; its error is returned to the agent for its next decision.")
            message = self._say("The file could not be read.", "File read nahi ho saki.") if tool == "read_file" else f"{self._tool_friendly_name(tool)} failed."
            self._set_activity(message, reason=f"Error: {error}", technical=f"{tool} error: {error}")
        elif tool == "run_command":
            stdout = str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))
            command = str(event.get("command", ""))
            if self._looks_like_gui_smoke(command):
                self.gui_status = "Command exited successfully; UI behavior not confirmed" if result.get("success") and result.get("return_code") == 0 else "Command failed"
            if any(word in command.lower() for word in ("pytest", "unittest", "test", "check")):
                if result.get("success") and result.get("return_code") == 0:
                    self._set_activity("The check command completed. Reviewing its actual output.", reason=stdout.strip()[-250:])
                else:
                    self._set_phase("repairing", "Repairing", "A check failed; the agent is reviewing the output for a fix.")
                    self._set_activity("The check failed. Its output is returned to the agent for its next decision.", reason=(stdout.strip()[-300:] or f"Exit code: {result.get('return_code')}"))

    def _handle_worker(self, event):
        worker_id = str(event.get("worker_id", "worker"))
        status = str(event.get("status", "QUEUED" if event["event"] == "worker.queued" else "RUNNING")).upper()
        old = self.workers.get(worker_id, {})
        worker_type = str(event.get("worker_type", old.get("worker_type", "tool")))
        item = {
            "task": str(event.get("task", old.get("task", "")))[:160],
            "status": status,
            "operation": str(event.get("operation", old.get("operation", ""))),
            "worker_type": worker_type,
            "duration": event.get("duration", old.get("duration")),
            "started_at": (old.get("started_at") or time.monotonic()) if status == "RUNNING" else old.get("started_at"),
            "summary": str(event.get("summary", old.get("summary", "")))[:240],
            "action": str(event.get("action", old.get("action", event.get("operation", "")))),
            "dependencies": event.get("dependencies", old.get("dependencies", [])),
            "blocked_reason": event.get("blocked_reason", old.get("blocked_reason", "")),
        }
        if event["event"] == "worker.blocked":
            item["status"] = "BLOCKED"
        self.workers[worker_id] = item
        active = sum(worker["status"] == "RUNNING" for worker in self.workers.values())
        self._set_activity(f"{active} parallel worker(s) active" if active else f"Worker {worker_id}: {item['status'].lower()}" ,
                           reason=item["summary"] or item["task"] or "Parallel read operation.")

    def _handle_file(self, event):
        path = str(event.get("path", ""))
        if path:
            self.current_file = path.replace("\\\\", "/")
        action = str(event.get("action", "read"))
        if action == "folder_created":
            self.created_folders.add(path)
            self._set_activity(f"Project folder created: {path}", reason="The folder tool reported success.")
            return
        prior = self.files.get(path, {})
        added, removed = event.get("added"), event.get("removed")
        if action == "deleted":
            removed = prior.get("lines") if prior.get("lines") is not None else removed
        actions = list(prior.get("actions", []))
        if action not in actions:
            actions.append(action)
        line_count = event.get("lines", prior.get("lines"))
        if action == "deleted":
            line_count = prior.get("lines", line_count)
        self.files[path] = {"action": action, "actions": actions, "lines": line_count,
                            "added": self._sum_change(prior.get("added"), added),
                            "removed": self._sum_change(prior.get("removed"), removed),
                            "summary": event.get("summary", self._file_summary(action))}
        self._set_activity(f"{self._file_summary(action)} {path}", reason="The file change was reported by an actual tool operation.")

    def _handle_verification(self, event):
        item = {"message": str(event.get("message", "Check recorded")),
                "success": event.get("success"), "kind": event.get("kind", "check"),
                "command": event.get("command"), "duration": event.get("duration")}
        self.verification.append(item)
        self.verification = self.verification[-20:]
        self._set_phase("testing", "Testing", "The agent recorded a check result.")
        status = "passed" if item["success"] is True or "PASS" in item["message"].upper() else "failed" if item["success"] is False or "FAIL" in item["message"].upper() else "recorded"
        self._set_activity(f"A verification check {status}: {item['message']}", reason="This status comes from an emitted verification event.")

    def _handle_report(self, event):
        report = event.get("report", {})
        if not isinstance(report, dict):
            return
        self.unresolved_issues = [str(item) for item in report.get("completeness_issues", [])]
        self.unresolved_issues.extend(str(report.get("initial_error")) for _ in [0] if report.get("initial_error"))
        self.verification_status = self._verification_label(report.get("verification_status", "NOT VERIFIED"))
        self.commands = [{"command": str(item.get("command", "Command")),
                          "status": "passed" if item.get("success") and item.get("return_code") == 0 else "failed",
                          "return_code": item.get("return_code"), "duration": item.get("duration_seconds"),
                          "test_counts": item.get("test_counts", {})}
                         for item in report.get("commands", [])]
        for command in self.commands:
            if command["status"] == "failed":
                issue = f"Command failed: {command['command']} (exit code {command.get('return_code', 'unknown')})"
                if issue not in self.unresolved_issues:
                    self.unresolved_issues.append(issue)
        for check in report.get("static_checks", []):
            if isinstance(check, dict) and not check.get("success"):
                path = check.get("path", "changed file")
                issue = f"File check failed for {path}: {check.get('error', 'details not reported')}"
                if issue not in self.unresolved_issues:
                    self.unresolved_issues.append(issue)
        for command in self.commands:
            if self._looks_like_gui_smoke(command["command"]):
                self.gui_status = "Command exited successfully; UI behavior not confirmed" if command["status"] == "passed" else "Command failed"
        if self.verification_status == "Verified":
            self.workflow["testing"] = "complete"
            self.workflow["verifying"] = "complete"
        elif self.verification_status == "Not verified":
            if not self.commands:
                self.workflow["testing"] = "skipped"
            self.workflow["verifying"] = "failed"

    def _set_phase(self, phase, state="active", explanation=""):
        aliases = {"Understand request": "understanding", "Inspect workspace": "inspecting", "Plan": "planning",
                   "Implement": "implementing", "Test": "testing", "Repair errors": "repairing", "Verify result": "verifying"}
        key = aliases.get(phase, phase)
        if key not in self.workflow:
            return
        state = str(state).lower()
        if state == "active":
            for previous, _label in self.WORKFLOW:
                if previous == key:
                    break
                if self.workflow[previous] == "active":
                    self.workflow[previous] = "complete"
            self.workflow[key] = "active"
        elif state in {"complete", "completed", "done"}:
            self.workflow[key] = "complete"
        elif state in {"failed", "error"}:
            self.workflow[key] = "failed"
        elif state == "skipped":
            self.workflow[key] = "skipped"
        label = dict(self.WORKFLOW).get(key, key.title())
        self.phase = f"{label}" + (" · in progress" if self.workflow[key] == "active" else "")
        if explanation:
            self.explanation = str(explanation)

    def _set_activity(self, message, reason="", technical=""):
        self.activity = redact_secrets(str(message))
        self.current_reason = redact_secrets(str(reason))
        item = {"message": self.activity, "reason": str(reason), "time": time.monotonic()}
        if technical:
            item["technical"] = redact_secrets(str(technical))
        if not self.activity_log or self.activity_log[-1]["message"] != item["message"]:
            self.activity_log.append(item)
            self.activity_log = self.activity_log[-10:]
        if technical:
            self.technical_log.append(redact_secrets(f"{self.activity} | {technical}"))
            self.technical_log = self.technical_log[-30:]

    def _technical_event(self, event):
        if event["event"] in {"activity", "step.started", "tool.started", "tool.finished", "model.request", "model.response"}:
            detail = {key: value for key, value in event.items() if key not in {"event", "timestamp"}}
            self.technical_log.append(redact_secrets(f"{event['event']}: {detail}"))
            self.technical_log = self.technical_log[-30:]

    def _refresh(self):
        if self.live:
            # Let Rich's refresh timer batch changes and avoid a redraw per low-level event.
            self.live.update(self._render(), refresh=False)
        elif self.started_at is not None and self.activity:
            if self.activity != self._last_printed_activity:
                print(f"  {self.activity}", file=self.stream, flush=True)
                self._last_printed_activity = self.activity

    def _render(self):
        if not self.rich_enabled:
            return self.activity

        project = os.path.basename(os.path.normpath(self.workspace)) or self.workspace
        header = Table.grid(expand=True, padding=(0, 1))
        header.add_column(width=3)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(
            Text("▣", style="bold bright_cyan"),
            Group(
                Text("LOCAL CODEX", style="bold white"),
                Text(f"{self.provider.title()} · {self.model} · {self.execution_mode.title()}", style="dim"),
            ),
            Text(self._build_badge(), style=f"bold {self._build_badge_style()}"),
        )

        title = Text(self._say("What's happening?", "अभी क्या हो रहा है?"), style="bright_cyan")
        task = Text(self._task_headline(), style="bold white")
        rows = Table.grid(expand=True, padding=(0, 1))
        rows.add_column(width=3)
        rows.add_column(ratio=1)
        rows.add_column(justify="right")
        for label, state in self._visual_checklist():
            mark = "✓" if state == "done" else "!" if state == "failed" else "◉" if state == "active" else "●" if state == "skipped" else "○"
            style = {"done": "bright_cyan", "active": "bright_cyan", "failed": "red", "skipped": "dim", "pending": "dim"}.get(state, "white")
            status = "Done" if state == "done" else "Running" if state == "active" else "Failed" if state == "failed" else "Skipped" if state == "skipped" else ""
            rows.add_row(Text(mark, style=style), Text(label, style="white" if state != "pending" else "dim"),
                         Text(status, style=style))

        body_items = [title, Text(""), task, Text(""), rows, Text("")]
        if self.current_file:
            body_items.extend([
                Text("─" * 48, style="dim"),
                Text("‹/›  Current file: ", style="bright_cyan") + Text(self.current_file, style="bold white on grey15"),
            ])
        body_items.extend([Text(""), Text(f"Result: {self._result_label()}", style="bold")])
        if self.last_error:
            body_items.extend([Text(""), Text(f"Issue / recovery: {self.last_error[:180]}", style="yellow")])
        body = Group(*body_items)
        sections = [
            header,
            Panel(body, border_style="grey30", padding=(1, 1), expand=True),
        ]
        if self.show_technical and self.technical_log:
            sections.append(Panel("\n".join(self._escape(line) for line in self.technical_log[-6:]),
                                  title="Debug log", border_style="white"))
        return Group(*sections)

    def _build_badge(self):
        if self.task_status.startswith("Finished"):
            return "FAILED" if self.verification_status in {"Failed", "Blocked"} else "DONE"
        if self.started_at is not None:
            return "BUILDING"
        return "READY"

    def _build_badge_style(self):
        badge = self._build_badge()
        return "red" if badge == "FAILED" else "bright_green" if badge in {"BUILDING", "DONE"} else "dim"

    def _task_headline(self):
        if self.task_status.startswith("Finished"):
            return self._say("Task finished.", "Kaam poora hua.")
        if self.workflow.get("testing") == "active":
            return self._say("Application test is running…", "Application test chal raha hai…")
        if self.workflow.get("repairing") == "active":
            return self._say("Error fix kar raha hoon…", "Error fix kar raha hoon…")
        if self.current_file and self.workflow.get("implementing") == "active":
            return self._say(f"Writing code in {self.current_file}…", f"{self.current_file} mein code likh raha hoon…")
        return self.activity or self._say("Ready for your task.", "Aapke task ke liye ready hoon.")

    def _visual_checklist(self):
        base = dict(self._checklist())
        changed_python = any(
            path.lower().endswith((".py", ".pyw"))
            and any(action in {"created", "modified", "modified via command"} for action in item["actions"])
            for path, item in self.files.items()
        )
        writing = self.workflow.get("implementing") == "active"
        testing = next((item for item in reversed(self.commands)
                        if any(word in item["command"].lower() for word in ("pytest", "unittest", "test", "python "))), None)
        test_state = "done" if testing and testing["status"] == "passed" else "failed" if testing and testing["status"] == "failed" else "active" if self.workflow.get("testing") == "active" else "pending"
        folder_state = "done" if base.get("Project folder: created") == "done" else "active" if self.workflow.get("implementing") == "active" else "pending"
        python_state = "done" if changed_python else "active" if writing and self.current_file.lower().endswith((".py", ".pyw")) else "pending"
        code_state = "done" if changed_python and self.verification_status == "Verified" else "failed" if self.workflow.get("implementing") == "failed" else "active" if writing or changed_python else "pending"
        return [
            (self._say("Project folder banao", "Project folder banaya"), folder_state),
            (self._say("Python files banao", "Python files banाईं"), python_state),
            (self._say("Code likh raha hoon", "Code likh raha hoon"), code_state),
            (self._say("Application test kar raha hoon", "Application test kar raha hoon"), test_state),
        ]

    def _checklist(self):
        changed = [item for item in self.files.values() if any(action in {"created", "modified", "modified via command", "deleted"} for action in item["actions"])]
        folder = bool(self.created_folders) or any(
            "/" in path and item.get("action") in {"created", "modified"} for path, item in self.files.items())
        tests = [item for item in self.commands if any(word in item["command"].lower() for word in ("pytest", "unittest", "test"))]
        test_state = {"passed": "done", "failed": "failed"}.get(tests[-1]["status"], "pending") if tests else "pending"
        impl_state = "done" if changed and self.verification_status == "Verified" else "active" if changed else "pending"
        verify_state = "done" if self.verification_status == "Verified" else "failed" if self.verification_status in {"Failed", "Blocked"} else "pending"
        return [("Project folder: created" if folder else "Project folder: not reported", "done" if folder else "pending"),
                (f"Files changed: {len(changed)}", "done" if changed else "pending"),
                ("Implementation", impl_state), (f"Testing: {'passed' if test_state == 'done' else 'failed' if test_state == 'failed' else 'not run'}", test_state),
                (f"Verification: {self.verification_status}", verify_state)]

    def _result_label(self):
        if self.verification_status == "Verified": return "Completed and verified"
        if self.verification_status == "Blocked": return "Blocked"
        if self.verification_status == "Failed": return "Failed"
        if self.task_status.startswith("Finished"): return "Completed with warnings"
        return "In progress" if self.started_at else "Ready"

    def _plain_summary(self):
        changed = sum(any(action in {"created", "modified", "modified via command", "deleted"}
                          for action in item["actions"]) for item in self.files.values())
        tests = next((label.split(": ", 1)[1] for label, _ in self._checklist()
                      if label.startswith("Testing:")), "not run")
        lines = [f"Result: {self._result_label()} ({self.verification_status})",
                 f"Files changed: {changed}", f"Tests: {tests}"]
        if self.last_error:
            lines.append(f"Issue / recovery: {self.last_error[:180]}")
        elif self.unresolved_issues:
            lines.append(f"Issue: {self.unresolved_issues[0]}")
        if self.show_technical and self.technical_log:
            lines.append("Debug: " + self.technical_log[-1])
        return "\n".join(lines)

    def _say(self, english, hinglish):
        return hinglish if self.demo_mode and self.language == "hinglish" else english

    def _tool_description(self, tool, arguments, old_label):
        path = str(arguments.get("path", "")) if isinstance(arguments, dict) else ""
        if path and tool in {"write_file", "edit_file", "read_file"}:
            self.current_file = path.replace("\\\\", "/")
        query = str(arguments.get("query", "")) if isinstance(arguments, dict) else ""
        command = str(arguments.get("command", "")) if isinstance(arguments, dict) else ""
        hinglish = self.demo_mode and self.language == "hinglish"
        descriptions = {
            "list_files": ("Project ki files inspect kar raha hai." if hinglish else "Looking through the project files.", "This helps the agent understand the existing project."),
            "read_file": (f"Project file padh raha hai: {path}." if hinglish else f"Reading project file: {path}.", "The agent needs the existing code before deciding what to change."),
            "write_file": (f"Nayi file bana raha hai: {path}." if hinglish else f"Creating a project file: {path}.", "The requested implementation needs this file."),
            "edit_file": (f"File update kar raha hai: {path}." if hinglish else f"Updating project file: {path}.", "The agent is applying a change to existing code."),
            "search_files": (f"Project mein dhoondh raha hai: {query}." if hinglish else f"Searching the project for: {query}.", "This helps locate relevant code before editing."),
            "run_command": ((f"Project command chala raha hai: {command}." if hinglish else f"Running project command: {command}."), "The command checks, builds, or runs the application; its actual result will follow."),
            "create_folder": (f"Project folder bana raha hai: {path}." if hinglish else f"Creating project folder: {path}.", "The files need a location inside the project."),
        }
        return descriptions.get(tool, (old_label or f"Running {tool}.", "The agent requested this tool operation."))

    def _tool_friendly_name(self, tool):
        return {"list_files": "Project file inspection", "read_file": "File read", "write_file": "File creation",
                "edit_file": "File update", "search_files": "Project search", "run_command": "Project command",
                "create_folder": "Folder creation"}.get(tool, "Tool operation")

    def _friendly_status(self, text):
        lowered = text.lower()
        if "repeated action" in lowered:
            return "The agent repeated an action, so it is trying a different way to continue."
        if "failed" in lowered or "error" in lowered:
            return "The agent found a problem and is checking how to recover."
        if "waiting for" in lowered and "model" in lowered:
            return "The AI is choosing its next action."
        return text if not self.demo_mode else ("Agent kaam kar raha hai." if self.language == "hinglish" else text)

    def _friendly_recovery(self, text):
        if "repeated action" in text.lower():
            return "The agent repeated an action, so it is trying a different way to continue."
        if "step budget" in text.lower():
            return "The agent is trying to finish missing files before stopping."
        return "The agent found a problem and is trying a recovery."

    def _reason_for_status(self, text):
        lowered = text.lower()
        if "inspect" in lowered:
            return "Existing project context is needed before making changes."
        if "test" in lowered or "verif" in lowered:
            return "The agent is checking whether its changes work."
        if "repair" in lowered or "recover" in lowered:
            return "A previous operation or check reported a problem."
        return "Status reported by the agent task loop."

    @staticmethod
    def _file_summary(action):
        return {"created": "Created", "modified": "Updated", "modified via command": "Updated by command",
                "deleted": "Deleted", "read": "Read"}.get(action, action.title())

    @staticmethod
    def _sum_change(previous, current):
        if previous is None and current is None:
            return None
        if current is None:
            return previous
        return (previous or 0) + current

    @staticmethod
    def _duration(value):
        if value is None:
            return "not reported"
        try:
            return f"{max(0, float(value)):.1f}s"
        except (TypeError, ValueError):
            return "not reported"

    @staticmethod
    def _verification_label(value):
        return {"PASS": "Verified", "READ ONLY": "Answer only - no changes verified",
                "PARTIAL": "Warnings", "FAIL": "Failed", "FAILED": "Failed",
                "BLOCKED": "Blocked"}.get(str(value).upper(), "Not verified")

    @staticmethod
    def _looks_like_gui_smoke(command):
        command = str(command).lower()
        return any(word in command for word in ("smoke", "launch", "gui", "tkinter", "pygame"))

    @staticmethod
    def _escape(text):
        return str(text).replace("[", "\\[")
