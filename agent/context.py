"""Compact, workspace-scoped repository context and targeted source retrieval."""

import ast
import fnmatch
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time

from agent.orchestration import redact_secrets


IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea",
    ".vscode", ".localcodex", "dist", "build", "target", "coverage", ".next", "vendor",
}
SENSITIVE_NAMES = {".env", ".env.local", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".wav",
    ".mp3", ".ogg", ".mp4", ".mov", ".pdf", ".zip", ".gz", ".7z",
    ".exe", ".dll", ".so", ".pyc", ".pyd", ".db", ".sqlite", ".bin", ".pem", ".key",
}
MANIFESTS = {
    "pyproject.toml", "requirements.txt", "requirements-dev.txt", "pipfile",
    "pipfile.lock", "poetry.lock", "uv.lock", "package.json", "package-lock.json",
    "pnpm-lock.yaml", "yarn.lock", "cargo.toml", "cargo.lock", "go.mod", "go.sum",
    "pom.xml", "build.gradle", "build.gradle.kts", "gemfile", "composer.json",
    "makefile", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
}
ENTRYPOINT_NAMES = {"main.py", "app.py", "run.py", "manage.py", "main.js", "index.js", "app.js"}
MAX_INDEX_FILES = 3000
MAX_INDEX_DEPTH = 10


def _path_key(path):
    return path.replace("\\", "/").lower()


class WorkspaceContextManager:
    """Refreshable index which reads only compact, task-relevant source excerpts."""

    def __init__(self, workspace, max_context_chars=14000):
        self.workspace = os.path.realpath(os.path.abspath(workspace))
        self.max_context_chars = max(4000, int(max_context_chars))
        self._files = None
        self._git_cache = None
        self._git_cached_at = 0.0
        self._text_cache = {}

    def invalidate(self, changed_paths=None):
        """Drop inventory and git metadata after writes, edits, deletes, or commands."""
        self._files = None
        self._git_cache = None
        self._git_cached_at = 0.0
        if changed_paths is None:
            self._text_cache.clear()
        else:
            for path in changed_paths:
                self._text_cache.pop(_path_key(str(path)), None)

    def _git_files(self):
        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=self.workspace, capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode:
            return None
        return [path for path in result.stdout.split("\0") if path][:MAX_INDEX_FILES]

    def _fallback_files(self):
        files = []
        inherited_rules = {".": []}

        def ignored(path, rules):
            state = False
            for base, source in rules:
                negate = source.startswith("!")
                pattern = source[1:] if negate else source
                if not pattern:
                    continue
                relative = path if base == "." else os.path.relpath(os.path.join(self.workspace, path), os.path.join(self.workspace, base)).replace("\\", "/")
                if relative == ".." or relative.startswith("../"):
                    continue
                directory_pattern = pattern.endswith("/")
                pattern = pattern.strip("/")
                matched = (
                    relative == pattern or relative.startswith(pattern + "/")
                    if directory_pattern else
                    fnmatch.fnmatch(relative, pattern) if "/" in pattern else
                    any(fnmatch.fnmatch(part, pattern) for part in relative.split("/"))
                )
                if matched:
                    state = not negate
            return state

        for root, dirs, names in os.walk(self.workspace):
            relative_root = os.path.relpath(root, self.workspace)
            depth = 0 if relative_root == "." else relative_root.count(os.sep) + 1
            if depth >= MAX_INDEX_DEPTH or len(files) >= MAX_INDEX_FILES:
                dirs[:] = []
                continue
            rules = list(inherited_rules.get(relative_root, []))
            dirs[:] = sorted(
                name for name in dirs
                if name not in IGNORED_DIRS
            )
            ignore_file = os.path.join(root, ".gitignore")
            try:
                with open(ignore_file, "r", encoding="utf-8", errors="replace") as handle:
                    rules.extend(
                        (relative_root, line.strip()) for line in handle
                        if line.strip() and not line.lstrip().startswith("#")
                    )
            except OSError:
                pass
            for name in dirs:
                child = name if relative_root == "." else f"{relative_root}/{name}"
                inherited_rules[child] = rules
            dirs[:] = [name for name in dirs if not ignored(
                name if relative_root == "." else f"{relative_root}/{name}", rules
            )]
            for name in sorted(names):
                relative = name if relative_root == "." else os.path.join(relative_root, name)
                normalized = relative.replace("\\", "/")
                if name == ".gitignore" or self._excluded(normalized) or ignored(normalized, rules):
                    continue
                files.append(normalized)
                if len(files) >= MAX_INDEX_FILES:
                    dirs[:] = []
                    break
        return files

    @staticmethod
    def _excluded(path):
        parts = path.split("/")
        basename = parts[-1].lower()
        return (
            any(part in IGNORED_DIRS for part in parts)
            or basename in SENSITIVE_NAMES
            or basename.startswith(".env")
            or basename.startswith(("secret", "credential"))
            or os.path.splitext(basename)[1] in BINARY_SUFFIXES
        )

    def files(self, refresh=False):
        if refresh:
            self._files = None
        if self._files is None:
            discovered = self._git_files()
            if discovered is None:
                discovered = self._fallback_files()
            valid = []
            for path in discovered:
                normalized = path.replace("\\", "/")
                if self._excluded(normalized):
                    continue
                target = os.path.realpath(os.path.join(self.workspace, path))
                try:
                    if os.path.commonpath([self.workspace, target]) != self.workspace or not os.path.isfile(target):
                        continue
                except ValueError:
                    continue
                valid.append(normalized)
            self._files = sorted(set(valid), key=str.casefold)
        return list(self._files)

    def _git_context(self):
        now = time.monotonic()
        if self._git_cache is not None and now - self._git_cached_at < 2:
            return self._git_cache
        status = recent = "unavailable"
        try:
            result = subprocess.run(
                ["git", "status", "--short", "--untracked-files=normal"], cwd=self.workspace,
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                status = result.stdout.strip() or "clean"
            history = subprocess.run(
                ["git", "log", "-5", "--oneline"], cwd=self.workspace,
                capture_output=True, text=True, timeout=5, check=False,
            )
            if history.returncode == 0:
                recent = history.stdout.strip() or "no commits"
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._git_cache = (status[:1600], recent[:900])
        self._git_cached_at = now
        return self._git_cache

    def _read(self, relative, limit=5000):
        if self._excluded(relative):
            return ""
        target = os.path.realpath(os.path.join(self.workspace, relative))
        try:
            stat = os.stat(target)
            if os.path.commonpath([self.workspace, target]) != self.workspace or stat.st_size > 1_000_000:
                return ""
            key = _path_key(relative)
            cached = self._text_cache.get(key)
            signature = (stat.st_mtime_ns, stat.st_size)
            if cached and cached[0] == signature and cached[1] >= limit:
                return cached[2][:limit]
            with open(target, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read(min(1_000_000, max(limit, 6000)) + 1)
            self._text_cache[key] = (signature, len(content), redact_secrets(content))
            return self._text_cache[key][2][:limit]
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _tokens(text):
        return {token.casefold() for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)}

    def _rank(self, path, task_tokens, report_paths, report_errors):
        normalized = _path_key(path)
        parts = set(re.split(r"[/_.-]+", normalized))
        score = 3 * len(parts & task_tokens)
        score += 7 if normalized in report_paths else 0
        score += 5 if os.path.basename(normalized) in ENTRYPOINT_NAMES else 0
        score += 2 if "/test" in normalized or normalized.startswith("test_") else 0
        if report_errors:
            score += min(5, sum(1 for word in report_errors if word in normalized))
        return score

    @staticmethod
    def _python_outline(path, content):
        if not path.lower().endswith((".py", ".pyw")):
            return ""
        try:
            tree = ast.parse(content)
        except SyntaxError as error:
            return f"Structure: parse failed at line {error.lineno}: {error.msg}"
        imports, symbols = [], []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(f"{node.__class__.__name__} {node.name} (line {node.lineno})")
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append("." * node.level + (node.module or ""))
        parts = []
        if imports:
            parts.append("Imports: " + ", ".join(dict.fromkeys(imports))[:500])
        if symbols:
            parts.append("Definitions: " + "; ".join(symbols[:30]))
        return " | ".join(parts)

    def build_context(self, task="", previous_report=None, recent_history=None,
                      recent_tool_calls=None, specification=None, refresh=True):
        files = self.files(refresh=refresh)
        manifests = [path for path in files if os.path.basename(path).lower() in MANIFESTS]
        readmes = [path for path in files if os.path.basename(path).lower().startswith("readme")][:2]
        entrypoints = [path for path in files if os.path.basename(path).lower() in ENTRYPOINT_NAMES][:12]
        tests = [path for path in files if re.search(r"(^|/)(tests?|__tests__)/|(^|/)test_[^/]+$", path, re.I)][:20]
        extension_counts = {}
        for path in files:
            suffix = os.path.splitext(path)[1].lower()
            if suffix:
                extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
        languages = sorted(extension_counts, key=lambda suffix: (-extension_counts[suffix], suffix))[:8]
        if any(suffix in extension_counts for suffix in (".py", ".pyw")):
            project_type = "Python project"
        elif ".js" in extension_counts or ".ts" in extension_counts or ".tsx" in extension_counts:
            project_type = "JavaScript/TypeScript project"
        elif ".go" in extension_counts:
            project_type = "Go project"
        elif ".rs" in extension_counts:
            project_type = "Rust project"
        else:
            project_type = "Mixed or unclassified project"

        report = previous_report if isinstance(previous_report, dict) else {}
        report_paths = {_path_key(path) for path in report.get("changed_files", [])}
        report_paths.update(_path_key(path) for path in report.get("missing_artifacts", []))
        report_errors = self._tokens(" ".join([
            str(report.get("initial_error", "")),
            " ".join(str(item.get("stderr", "")) for item in report.get("commands", []) if isinstance(item, dict)),
            " ".join(str(item) for item in report.get("completeness_issues", [])),
        ]))
        task_tokens = self._tokens(task)
        task_tokens.update(report_errors & {"gui", "window", "import", "module", "asset", "test", "error", "failed", "traceback"})
        candidates = list(dict.fromkeys(manifests + readmes + entrypoints + tests))
        ranked = sorted(files, key=lambda path: self._rank(path, task_tokens, report_paths, report_errors), reverse=True)
        for path in ranked[:8]:
            if path not in candidates:
                candidates.append(path)

        status, recent_commits = self._git_context()
        test_commands = []
        dependency_status = []
        if tests and project_type == "Python project":
            test_commands = ["python -m pytest", "python -m unittest discover -s tests"]
        for manifest in manifests:
            if os.path.basename(manifest).lower() not in {"requirements.txt", "requirements-dev.txt"}:
                continue
            for requirement in self._read(manifest, 16000).splitlines():
                match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
                if not match or requirement.lstrip().startswith(("#", "-")):
                    continue
                package = match.group(1)
                try:
                    installed = importlib.metadata.version(package)
                    dependency_status.append(f"{package}: installed {installed}")
                except importlib.metadata.PackageNotFoundError:
                    dependency_status.append(f"{package}: MISSING from active Python environment")
                if len(dependency_status) >= 15:
                    break
        package_json = next((path for path in manifests if os.path.basename(path).lower() == "package.json"), None)
        if package_json:
            try:
                package = json.loads(self._read(package_json, 16000))
                scripts = package.get("scripts", {})
                test_commands.extend(
                    f"npm run {name}" for name in ("test", "lint", "build") if name in scripts
                )
            except (ValueError, AttributeError):
                pass
        missing_artifacts = []
        expected = ((report.get("specification") or {}).get("expected_files", [])
                    if isinstance(report.get("specification"), dict) else [])
        expected.extend(report.get("missing_artifacts", []))
        for item in expected:
            path = item.get("path") if isinstance(item, dict) else item
            if not path or os.path.isabs(path):
                continue
            target = os.path.realpath(os.path.join(self.workspace, path))
            try:
                if os.path.commonpath([self.workspace, target]) == self.workspace and not os.path.exists(target):
                    missing_artifacts.append(path)
            except ValueError:
                continue
        lines = [
            f"WORKSPACE ROOT: {self.workspace}",
            f"PROJECT: {project_type}; indexed files {len(files)}{' (inventory cap reached)' if len(files) >= MAX_INDEX_FILES else ''}; common file types {', '.join(languages) or 'none'}",
            "INDEX NOTES: generated directories and binaries are excluded; use targeted search tools for files outside this compact context.",
            f"RUNTIME: Python {sys.version_info.major}.{sys.version_info.minor}; executable {sys.executable}; current directory {os.getcwd()}",
            "DIRECTORY TREE (selected files): " + ", ".join(files[:100]),
            "ENTRY POINTS: " + (", ".join(entrypoints) or "not detected"),
            "TEST FILES: " + (", ".join(tests) or "not detected"),
            "CANDIDATE TEST COMMANDS (not yet executed): " + ("; ".join(test_commands) or "not detected"),
            "DEPENDENCY/PROJECT MANIFESTS: " + (", ".join(manifests) or "not detected"),
            "ACTIVE PYTHON DEPENDENCIES: " + ("; ".join(dependency_status) or "no requirements.txt dependencies detected"),
            "GIT STATUS: " + status,
            "RECENT COMMITS: " + recent_commits,
        ]
        if specification:
            lines.append("CURRENT TASK PLAN: " + redact_secrets(str(specification))[:1800])
        if report:
            report_summary = {
                key: report.get(key) for key in (
                    "task", "changed_files", "commands", "static_checks", "verification_status",
                    "initial_error", "completeness_issues", "missing_artifacts", "escalations",
                ) if report.get(key) is not None
            }
            lines.append("PREVIOUS TASK EXECUTION REPORT: " + redact_secrets(str(report_summary))[:2300])
        if missing_artifacts:
            lines.append("MISSING REQUIRED ARTIFACTS (current filesystem check): " + ", ".join(dict.fromkeys(missing_artifacts)))
        turns = (recent_history or [])[-4:]
        if turns:
            summary = [f"{item.get('role')}: {redact_secrets(str(item.get('content', ''))[:450])}" for item in turns]
            lines.append("RECENT CONVERSATION: " + " | ".join(summary))
        calls = (recent_tool_calls or [])[-5:]
        if calls:
            lines.append("RECENT TOOL OUTCOMES: " + redact_secrets(str(calls))[:2200])

        room = self.max_context_chars - sum(len(item) + 1 for item in lines)
        for path in candidates:
            if room <= 500:
                break
            content = self._read(path, min(3000, max(500, room - 250)))
            if not content:
                target = os.path.realpath(os.path.join(self.workspace, path))
                try:
                    reason = "file exceeds the 1 MB context read limit" if os.path.getsize(target) > 1_000_000 else "file could not be read"
                except OSError:
                    reason = "file could not be read"
                note = f"\nFILE {path}: not included because {reason}."
                if len(note) < room:
                    lines.append(note)
                    room -= len(note)
                continue
            outline_source = self._read(path, 1_000_000) if path.lower().endswith((".py", ".pyw")) else content
            outline = self._python_outline(path, outline_source)
            block = f"\nFILE {path}:\n{outline}\n{content}" if outline else f"\nFILE {path}:\n{content}"
            if len(block) > room:
                block = block[:room] + "\n...[context excerpt truncated]"
            lines.append(block)
            room -= len(block)
        return redact_secrets("\n".join(lines)[:self.max_context_chars])


def build_project_context(workspace, task=""):
    """Backward-compatible compact project summary for callers using the old helper."""
    return WorkspaceContextManager(workspace).build_context(task)
