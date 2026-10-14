"""Persistent, evidence-backed workspace metadata kept inside each project."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading

from agent.orchestration import redact_secrets
from training.trajectory import build_trajectory_record, write_trajectory_jsonl


SCHEMA_VERSION = 1
MAX_INDEXED_FILE_BYTES = 512_000
MAX_TOTAL_INDEX_BYTES = 24_000_000
MEMORY_FILES = (
    "project_summary.md", "architecture.json", "file_index.json", "commands.json",
    "decisions.json", "active_tasks.json", "unresolved_issues.json",
    "recent_changes.json", "memory_manifest.json", "trajectories.jsonl",
)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_id(path):
    return hashlib.sha256(os.path.normcase(os.path.realpath(path)).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class WorkspaceMemory:
    """Incremental project index and compact task history with safe persistence."""

    def __init__(self, workspace, context_manager, config=None):
        self.workspace = os.path.realpath(os.path.abspath(workspace))
        if not os.path.isdir(self.workspace):
            raise ValueError("Workspace root must be an existing directory")
        self.context_manager = context_manager
        self.config = config or {}
        self.enabled = bool(self.config.get("persistent_memory_enabled", True))
        configured_dir = self.config.get("workspace_memory_dir", ".localcodex/memory")
        candidate = os.path.realpath(os.path.join(self.workspace, configured_dir))
        try:
            inside = os.path.commonpath([self.workspace, candidate]) == self.workspace
        except ValueError:
            inside = False
        self.enabled = self.enabled and inside
        self.directory = Path(candidate)
        self.workspace_id = _stable_id(self.workspace)
        self.lock = threading.RLock()
        self.status = "Disabled" if not self.enabled else "Not indexed"
        self.stale = False
        self.stale_detected = False
        self.last_refreshed = None
        self.retrieved_paths = []
        self._data = self._empty_data()
        if self.enabled:
            try:
                self._load()
            except OSError as error:
                self.enabled = False
                self.status = f"Disabled: memory storage unavailable ({type(error).__name__})"

    def _empty_data(self):
        return {
            "architecture": {"modules": [], "startup_flow": [], "data_flow": [], "status": "unknown"},
            "file_index": {"files": []}, "commands": {"items": []}, "decisions": {"items": []},
            "active_tasks": {"items": []}, "unresolved_issues": {"items": []}, "recent_changes": {"items": []},
            "manifest": {"schema_version": SCHEMA_VERSION, "workspace_id": self.workspace_id,
                         "workspace_path": self.workspace, "status": "new"},
            "project_summary": "Project summary has not been generated yet.",
        }

    def _load_json(self, name, default):
        path = self.directory / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else default
        except (OSError, UnicodeError, json.JSONDecodeError):
            if path.exists():
                self.stale = True
            return default

    def _load(self):
        if self.directory.exists() and any(self.directory.iterdir()):
            manifest_path = self.directory / "memory_manifest.json"
            if not manifest_path.exists() and any((self.directory / name).exists() for name in MEMORY_FILES):
                self.enabled = False
                self.status = "Disabled: existing memory files have no ownership manifest"
                return
        self.directory.mkdir(parents=True, exist_ok=True)
        trajectory_path = self.directory / "trajectories.jsonl"
        if not trajectory_path.exists():
            _atomic_write(trajectory_path, "")
        manifest = self._load_json("memory_manifest.json", self._data["manifest"])
        if (self.directory / "memory_manifest.json").exists() and not self.stale and manifest.get("workspace_id") != self.workspace_id:
            self.enabled = False
            self.status = "Disabled: memory manifest belongs to another application or workspace"
            return
        if manifest.get("workspace_id") not in (None, self.workspace_id):
            self.status = "Workspace mismatch; rebuilding memory"
            self.stale = True
            return
        self._data["manifest"] = manifest
        for key, filename in (
            ("architecture", "architecture.json"), ("file_index", "file_index.json"),
            ("commands", "commands.json"), ("decisions", "decisions.json"),
            ("active_tasks", "active_tasks.json"), ("unresolved_issues", "unresolved_issues.json"),
            ("recent_changes", "recent_changes.json"),
        ):
            loaded = self._load_json(filename, self._data[key])
            if any(not isinstance(loaded.get(field), type(default)) for field, default in self._data[key].items()):
                loaded = self._data[key]
                self.stale = True
            self._data[key] = loaded
        summary_path = self.directory / "project_summary.md"
        try:
            self._data["project_summary"] = summary_path.read_text(encoding="utf-8")[:6000]
        except (OSError, UnicodeError):
            if summary_path.exists():
                self.stale = True
        prior_path = self._data["manifest"].get("workspace_path")
        self.stale |= bool(prior_path and os.path.normcase(os.path.realpath(prior_path)) != os.path.normcase(self.workspace))
        self.status = "Loaded; refresh required" if self.stale else "Loaded"

    def _save_json(self, name, value):
        serialized = json.dumps(value, ensure_ascii=False, indent=2)
        _atomic_write(self.directory / name, redact_secrets(serialized) + "\n")

    def _index_files(self):
        previously_indexed = bool(self._data["manifest"].get("project_fingerprint"))
        was_stale = self.stale
        old = {item.get("path"): item for item in self._data["file_index"].get("files", []) if isinstance(item, dict)}
        files = self.context_manager.files(refresh=True)
        indexed, consumed = [], 0
        changed = []
        for relative in files:
            target = os.path.realpath(os.path.join(self.workspace, relative))
            try:
                if os.path.commonpath([self.workspace, target]) != self.workspace:
                    continue
                stat = os.stat(target)
            except (OSError, ValueError):
                continue
            entry = {"path": relative.replace("\\", "/"), "type": Path(relative).suffix.lower() or "directory",
                     "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns,
                     "status": "indexed", "symbols": [], "description": ""}
            prior = old.get(entry["path"])
            if prior and prior.get("modified_ns") == stat.st_mtime_ns and prior.get("size_bytes") == stat.st_size and prior.get("sha256"):
                entry.update({key: prior[key] for key in ("sha256", "symbols", "description", "status") if key in prior})
                consumed += stat.st_size
            elif stat.st_size > MAX_INDEXED_FILE_BYTES or consumed + stat.st_size > MAX_TOTAL_INDEX_BYTES:
                entry["status"] = "metadata_only"
                entry["description"] = "Content omitted because the configured indexing size limit was reached."
                changed.append(entry["path"])
            else:
                try:
                    with open(target, "rb") as stream:
                        raw = stream.read(MAX_INDEXED_FILE_BYTES + 1)
                    if len(raw) > MAX_INDEXED_FILE_BYTES or b"\x00" in raw[:4096]:
                        entry["status"] = "metadata_only"
                        entry["description"] = "Binary or oversized content was not read."
                    else:
                        content = raw.decode("utf-8", errors="replace")
                        consumed += len(raw)
                        entry["sha256"] = hashlib.sha256(raw).hexdigest()
                        entry["description"] = self._describe(entry["path"], content)
                        entry["symbols"] = self._symbols(entry["path"], content)
                        changed.append(entry["path"])
                except OSError:
                    entry["status"] = "unavailable"
            indexed.append(entry)
        current_paths = {item["path"] for item in indexed}
        deleted = sorted(set(old) - current_paths)
        self._data["file_index"] = {"files": indexed}
        self._data["architecture"] = self._architecture(indexed)
        self._data["project_summary"] = self._summary(indexed)
        self._refresh_discovered_commands(indexed)
        extension_counts = {}
        for item in indexed:
            suffix = item["type"]
            extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        if any(suffix in extension_counts for suffix in (".py", ".pyw")):
            project_type = "Python application"
        elif any(suffix in extension_counts for suffix in (".js", ".ts", ".tsx", ".html")):
            project_type = "Web application"
        elif ".go" in extension_counts:
            project_type = "Go project"
        else:
            project_type = "Mixed or unclassified project"
        self._data["manifest"] = {
            "schema_version": SCHEMA_VERSION, "workspace_id": self.workspace_id,
            "workspace_path": self.workspace, "last_full_scan": _now(),
            "last_incremental_scan": _now(),
            "project_fingerprint": hashlib.sha256("\n".join(f"{x['path']}:{x.get('sha256', x['modified_ns'])}" for x in indexed).encode()).hexdigest(),
            "indexing_status": "complete", "indexed_files": len(indexed),
            "changed_files": changed[-100:], "deleted_files": deleted[-100:],
            "stale_entries": [], "project_type": project_type,
        }
        self.stale_detected = bool(was_stale or (previously_indexed and (changed or deleted)))
        self.stale = False
        self.last_refreshed = self._data["manifest"]["last_incremental_scan"]
        self.status = "Refreshed" if changed or deleted or self.stale else "Current"
        if changed or deleted or not previously_indexed or was_stale:
            self._persist()
        return changed, deleted

    def _refresh_discovered_commands(self, indexed):
        existing = self._data["commands"].get("items", [])
        executed = [item for item in existing if item.get("source") == "executed"]
        discovered = {}
        for entry in indexed:
            if not Path(entry["path"]).name.lower().startswith("readme"):
                continue
            content = self.context_manager._read(entry["path"], 12000)
            pattern = r"(?<![\w])((?:python(?:\s+-m)?|py\s+-m|npm\s+(?:run|test)|pytest|cargo\s+test|go\s+test|mvn\s+test)\s+[^\r\n`]+|pytest\b[^\r\n`]*)"
            for match in re.finditer(pattern, content, re.I):
                candidate = match.group(1).strip().strip("`> *.")
                candidate = re.split(r"\s{2,}|\s+#", candidate, maxsplit=1)[0][:300]
                discovered[candidate] = {"command": candidate, "purpose": "Mentioned in README; not verified by execution.",
                                         "working_directory": str(Path(entry["path"]).parent).replace(".", "", 1),
                                         "source": "discovered", "verification": "unknown"}
        self._data["commands"]["items"] = list(discovered.values()) + executed[-100:]

    @staticmethod
    def _describe(path, content):
        if Path(path).name.lower().startswith("readme"):
            lines = [line.strip(" #\t") for line in content.splitlines() if line.strip() and not line.lstrip().startswith("```")]
            return redact_secrets(" ".join(lines[:3]))[:240]
        if path.lower().endswith(".py"):
            try:
                module = ast.parse(content)
                return redact_secrets(ast.get_docstring(module) or "Python source module")[:240]
            except SyntaxError:
                return "Python source with a syntax error during indexing."
        return f"{Path(path).suffix[1:].upper() or 'Text'} project file"

    @staticmethod
    def _symbols(path, content):
        if not path.lower().endswith((".py", ".pyw")):
            return []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        result = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                result.append({"name": node.name, "kind": type(node).__name__, "line": node.lineno})
        return result[:100]

    def _architecture(self, indexed):
        modules = []
        for entry in indexed:
            if entry["type"] not in {".py", ".pyw"}:
                continue
            relative = entry["path"]
            source = self.context_manager._read(relative, 64_000)
            imports = []
            try:
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(("." * node.level) + (node.module or ""))
            except SyntaxError:
                pass
            modules.append({"path": relative, "responsibility": entry["description"],
                            "imports": list(dict.fromkeys(imports))[:40], "relationship_confidence": "inferred"})
        return {"modules": modules[:400], "startup_flow": [item["path"] for item in indexed if Path(item["path"]).name.lower() in {"main.py", "app.py", "run.py", "manage.py"}][:30],
                "data_flow": [], "status": "inferred from source imports; verify before relying"}

    def _summary(self, indexed):
        suffixes = {}
        for item in indexed:
            suffixes[item["type"]] = suffixes.get(item["type"], 0) + 1
        languages = ", ".join(f"{suffix[1:] or suffix} ({count})" for suffix, count in sorted(suffixes.items(), key=lambda pair: -pair[1])[:8]) or "unknown"
        entrypoints = [item["path"] for item in indexed if Path(item["path"]).name.lower() in {"main.py", "app.py", "run.py", "index.js", "app.js"}]
        readmes = [item for item in indexed if Path(item["path"]).name.lower().startswith("readme")]
        purpose = readmes[0]["description"] if readmes else "Purpose not established from a README."
        manifests = [item["path"] for item in indexed if Path(item["path"]).name.lower() in {"requirements.txt", "pyproject.toml", "package.json", "cargo.toml", "go.mod"}]
        tests = [item["path"] for item in indexed if re.search(r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]+", item["path"], re.I)]
        known_commands = self._data["commands"].get("items", [])[-12:]
        command_text = "; ".join(
            f"{item.get('command')} ({'executed: ' + str(item.get('return_code')) if item.get('source') == 'executed' else 'discovered, unverified'})"
            for item in known_commands if item.get("command")
        ) or "not detected"
        limitations = [item.get("description", "") for item in self._data["unresolved_issues"].get("items", []) if item.get("status") == "unresolved"]
        return "\n".join(["# Workspace project summary", f"Purpose: {purpose}", f"Languages/file types: {languages}",
                            f"Candidate entry points (filename based): {', '.join(entrypoints) or 'unknown'}", f"Manifests/dependencies: {', '.join(manifests) or 'unknown'}",
                            f"Tests: {', '.join(tests[:30]) or 'not detected'}", f"Known run/test commands: {command_text}",
                            f"Known current limitations: {'; '.join(limitations[:8]) or 'none recorded'}", f"Indexed files: {len(indexed)}",
                            "Evidence: README text and current filesystem metadata; architecture imports are inferred."])

    def _persist(self):
        try:
            for key, filename in (("architecture", "architecture.json"), ("file_index", "file_index.json"),
                                  ("commands", "commands.json"), ("decisions", "decisions.json"),
                                  ("active_tasks", "active_tasks.json"), ("unresolved_issues", "unresolved_issues.json"),
                                  ("recent_changes", "recent_changes.json"), ("manifest", "memory_manifest.json")):
                self._save_json(filename, self._data[key])
            _atomic_write(self.directory / "project_summary.md", redact_secrets(self._data["project_summary"]) + "\n")
        except OSError as error:
            self.status = f"Memory write failed: {type(error).__name__}"
            self.stale = True

    def refresh(self):
        if not self.enabled:
            return {"status": self.status, "indexed_files": 0, "changed": [], "deleted": []}
        with self.lock:
            try:
                changed, deleted = self._index_files()
                return {"status": self.status, "indexed_files": len(self._data["file_index"].get("files", [])),
                        "changed": changed, "deleted": deleted, "stale": self.stale,
                        "stale_detected": self.stale_detected}
            except Exception as error:
                self.status = f"Unavailable: {type(error).__name__}"
                self.stale = True
                self.stale_detected = True
                return {"status": self.status, "indexed_files": len(self._data["file_index"].get("files", [])),
                        "changed": [], "deleted": [], "stale": True}

    def record_task_start(self, task, task_id, specification=None):
        if not self.enabled:
            return
        with self.lock:
            tasks = self._data["active_tasks"].setdefault("items", [])
            for item in tasks:
                if item.get("status") == "active":
                    item["status"] = "interrupted"
                    item["verification_state"] = "unknown after process restart"
            tasks[:] = tasks[-9:]
            tasks.append({"task_id": task_id, "original_request": redact_secrets(task)[:2000],
                          "plan": (specification or {}).get("acceptance_criteria", []), "completed_steps": [],
                          "pending_steps": list((specification or {}).get("expected_files", [])),
                          "changed_files": [], "verification_state": "in progress", "status": "active",
                          "last_checkpoint": _now()})
            self._persist()

    def record_task_finish(self, report):
        if not self.enabled:
            return
        with self.lock:
            task_id = report.get("task_id")
            tasks = self._data["active_tasks"].setdefault("items", [])
            entry = next((item for item in reversed(tasks) if item.get("task_id") == task_id), None)
            if entry is None:
                entry = {"task_id": task_id, "original_request": redact_secrets(str(report.get("task", "")))[:2000]}
                tasks.append(entry)
            entry.update({"status": "complete" if report.get("verification_status") == "PASS" else "incomplete",
                          "verification_state": report.get("verification_status", "unknown"),
                          "completed_steps": report.get("changed_files", [])[:100], "pending_steps": report.get("completeness_issues", [])[:50],
                          "changed_files": report.get("changed_files", [])[:100], "last_checkpoint": _now()})
            issue_items = self._data["unresolved_issues"].setdefault("items", [])
            issue_items[:] = [item for item in issue_items if item.get("task_id") != task_id]
            changed_paths = set(report.get("changed_files", []))
            if report.get("verification_status") == "PASS" and changed_paths:
                for item in issue_items:
                    if changed_paths.intersection(item.get("affected_files", [])):
                        item["status"] = "resolved"
                        item["resolved_by_task_id"] = task_id
                        item["resolution_verified"] = True
            if report.get("verification_status") not in {"PASS", "READ ONLY"}:
                command_evidence = []
                for command in report.get("commands", [])[-5:]:
                    command_evidence.append({key: redact_secrets(str(command.get(key, "")))[:900]
                                             for key in ("command", "return_code", "success", "stdout", "stderr") if key in command})
                issue_items.append({"task_id": task_id, "description": redact_secrets(str(report.get("initial_error") or "; ".join(report.get("completeness_issues", [])) or "Task not verified"))[:1200],
                                    "affected_files": report.get("changed_files", [])[:40],
                                    "commands": command_evidence, "status": "unresolved", "next_action": "Reproduce or run the relevant verification command."})
            changes = self._data["recent_changes"].setdefault("items", [])
            changes.append({"task_id": task_id, "task": redact_secrets(str(report.get("task", "")))[:500],
                            "files": report.get("changed_files", [])[:100],
                            "verification": report.get("verification_status", "unknown"), "recorded_at": _now()})
            self._data["recent_changes"]["items"] = changes[-30:]
            self._data["project_summary"] = self._summary(self._data["file_index"].get("files", []))
            self._persist()
            trajectory_path = self.directory / "trajectories.jsonl"
            trajectories = []
            try:
                for line in trajectory_path.read_text(encoding="utf-8").splitlines()[-29:]:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        trajectories.append(item)
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
            fingerprint = self._data["manifest"].get("project_fingerprint")
            trajectories.append(build_trajectory_record(report, self.workspace_id, fingerprint))
            try:
                write_trajectory_jsonl(trajectory_path, trajectories[-30:])
            except OSError:
                self.status = "Memory saved; trajectory write failed"
            if self.config.get("training_data_collection_enabled", False):
                try:
                    from training.pipeline import RAW, append_record, trajectory as training_trajectory
                    append_record(RAW, training_trajectory(report))
                except Exception as error:
                    # Training-data collection is best effort and must never fail an agent task.
                    self.status = f"Memory saved; training collection failed ({type(error).__name__})"

    def record_command(self, record):
        if not self.enabled:
            return
        with self.lock:
            safe = {key: record.get(key) for key in ("command", "cwd", "return_code", "success", "duration_seconds", "timed_out", "executed_at", "summary") if key in record}
            output = (str(record.get("stdout", "")) + "\n" + str(record.get("stderr", ""))).strip()
            if output:
                safe["summary"] = redact_secrets(output[-800:])
            safe["purpose"] = "Executed by EmiCoder"
            safe["source"] = "executed"
            items = self._data["commands"].setdefault("items", [])
            items.append(safe)
            self._data["commands"]["items"] = items[-100:]
            self._persist()

    def context_for(self, task, max_chars=4200, refresh=True):
        if not self.enabled:
            return "WORKSPACE MEMORY: disabled"
        max_chars = max(500, min(int(max_chars), 12000))
        if refresh:
            self.refresh()
        tokens = set(re.findall(r"[\w-]{2,}", str(task).casefold()))
        entries = self._data["file_index"].get("files", [])
        ranked = sorted(entries, key=lambda item: (sum(token in item.get("path", "").casefold() or token in item.get("description", "").casefold() for token in tokens), item.get("path", "")), reverse=True)
        related = [item for item in ranked if any(token in item.get("path", "").casefold() or token in item.get("description", "").casefold() or any(token in symbol.get("name", "").casefold() for symbol in item.get("symbols", [])) for token in tokens)][:12]
        self.retrieved_paths = [item["path"] for item in related]
        follow_up = bool(tokens & {"this", "that", "it", "same", "again", "error", "fails", "failed", "issue", "isme", "isko", "ab", "pichla"})
        def matches(item):
            material = json.dumps(item, ensure_ascii=False).casefold()
            return any(token in material for token in tokens)
        issue_items = [item for item in self._data["unresolved_issues"].get("items", []) if item.get("status") == "unresolved"]
        change_items = self._data["recent_changes"].get("items", [])
        command_items = self._data["commands"].get("items", [])
        issues = [item for item in issue_items if matches(item)][-4:]
        changes = [item for item in change_items if matches(item)][-4:]
        commands = [item for item in command_items if matches(item)][-4:]
        if follow_up:
            issues = (issues or issue_items[-1:])[-4:]
            changes = (changes or change_items[-1:])[-4:]
            commands = (commands or command_items[-2:])[-4:]
        payload = {
            "workspace_id": self.workspace_id[:16], "memory_status": self.status,
            "project_summary": self._data["project_summary"], "relevant_files": related,
            "architecture": [item for item in self._data["architecture"].get("modules", []) if item["path"] in self.retrieved_paths][:8],
            "recent_changes": changes, "unresolved_issues": issues, "recent_commands": commands,
            "trust": "Memory is a cache. Confirm relevant facts against the current filesystem before edits.",
        }
        serialized = redact_secrets(json.dumps(payload, ensure_ascii=False, default=str))
        return "PERSISTENT WORKSPACE MEMORY (retrieved for this request):\n" + serialized[:max_chars]

    def dashboard_state(self):
        active = next((item for item in reversed(self._data["active_tasks"].get("items", [])) if item.get("status") == "active"), None)
        return {"status": self.status, "indexed_files": len(self._data["file_index"].get("files", [])),
                "unresolved_issues": sum(1 for item in self._data["unresolved_issues"].get("items", []) if item.get("status") == "unresolved"),
                "last_task": (self._data["recent_changes"].get("items") or [{}])[-1].get("task", "none"),
                "stale": self.stale_detected, "context_files": len(self.retrieved_paths),
                "project_type": self._data["manifest"].get("project_type", "unknown"),
                "task_context_status": "active" if active else "loaded"}
