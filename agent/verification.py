"""Deterministic, non-executing checks for files written by Local Codex."""

import json
import os
import shutil
import subprocess


CODE_SUFFIXES = {
    ".py", ".pyw", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs",
}


def verify_changed_file(workspace, relative_path):
    """Check that a changed file exists, can be read back, and has parseable syntax when supported."""
    root = os.path.realpath(os.path.abspath(workspace))
    target = os.path.realpath(os.path.join(root, relative_path))
    try:
        if os.path.commonpath([root, target]) != root:
            return {
                "success": False,
                "path": relative_path,
                "checks": [],
                "errors": ["File path escapes the workspace."],
            }
    except ValueError:
        return {
            "success": False,
            "path": relative_path,
            "checks": [],
            "errors": ["File path escapes the workspace."],
        }

    checks = []
    errors = []
    if not os.path.isfile(target):
        return {
            "success": False,
            "path": relative_path,
            "checks": [],
            "errors": ["File does not exist after the write operation."],
        }
    checks.append("file exists")

    try:
        with open(target, "r", encoding="utf-8") as handle:
            content = handle.read()
    except (OSError, UnicodeError) as error:
        return {
            "success": False,
            "path": relative_path,
            "checks": checks,
            "errors": [f"Could not read file back: {error}"],
        }
    checks.append("file can be read back")

    suffix = os.path.splitext(target)[1].lower()
    if suffix in CODE_SUFFIXES and not content.strip():
        errors.append("Source file is empty.")

    if suffix in {".py", ".pyw"}:
        try:
            compile(content, target, "exec")
            checks.append("Python syntax valid")
        except (SyntaxError, ValueError) as error:
            errors.append(
                f"Python syntax error at line {getattr(error, 'lineno', '?')}: {error}"
            )
    elif suffix == ".json":
        try:
            json.loads(content)
            checks.append("JSON syntax valid")
        except json.JSONDecodeError as error:
            errors.append(f"JSON syntax error at line {error.lineno}, column {error.colno}: {error.msg}")
    elif suffix in {".js", ".mjs", ".cjs"}:
        node = shutil.which("node")
        if node:
            try:
                result = subprocess.run(
                    [node, "--check", target],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if result.returncode == 0:
                    checks.append("JavaScript syntax valid (node --check)")
                else:
                    errors.append((result.stderr or result.stdout or "node --check failed").strip()[:3000])
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(f"JavaScript syntax check failed to run: {error}")
        else:
            checks.append("JavaScript syntax check skipped (Node.js is not installed)")

    return {
        "success": not errors,
        "path": os.path.relpath(target, root),
        "checks": checks,
        "errors": errors,
    }
