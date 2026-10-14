"""Structured task specifications and task-aware deliverable checks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SPEC_FIELDS = {
    "task_type": str,
    "project_name": str,
    "intended_outcome": str,
    "language": str,
    "framework": str,
    "features": list,
    "expected_files": list,
    "dependencies": list,
    "run_commands": list,
    "test_commands": list,
    "acceptance_criteria": list,
    "assumptions": list,
    "definition_of_done": str,
}


class SpecificationError(ValueError):
    """Architect response is malformed or unsafe to use."""


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|token|password|secret)\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{8,})"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=?"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def redact_secrets(text: str) -> str:
    """Mask common credential forms before sending task context to a provider."""
    value = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 2:
            value = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED_SECRET]", value)
    return value


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SpecificationError("Architect response did not contain a JSON object")


def validate_specification(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecificationError("Specification must be an object")
    missing = [key for key in SPEC_FIELDS if key not in value]
    if missing:
        raise SpecificationError("Specification is missing fields: " + ", ".join(missing))
    for key, expected_type in SPEC_FIELDS.items():
        if not isinstance(value[key], expected_type):
            raise SpecificationError(f"Specification field {key!r} must be {expected_type.__name__}")
    for key in ("task_type", "project_name", "intended_outcome", "language", "definition_of_done"):
        if not value[key].strip():
            raise SpecificationError(f"Specification field {key!r} cannot be empty")
    for key in ("features", "dependencies", "run_commands", "test_commands", "acceptance_criteria", "assumptions"):
        if any(not isinstance(item, str) or not item.strip() for item in value[key]):
            raise SpecificationError(f"Specification field {key!r} must contain non-empty strings")

    files = []
    for item in value["expected_files"]:
        if isinstance(item, str):
            item = {"path": item, "responsibility": "Required project deliverable"}
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("responsibility"), str):
            raise SpecificationError("expected_files entries need path and responsibility strings")
        path = item["path"].replace("\\", "/").strip()
        parts = Path(path).parts
        if not path or path.startswith("/") or re.match(r"^[A-Za-z]:", path) or ".." in parts or "\x00" in path:
            raise SpecificationError(f"Unsafe expected file path: {path!r}")
        files.append({"path": path, "responsibility": item["responsibility"].strip()})
    value = dict(value)
    value["expected_files"] = files
    return value


def parse_specification(text: str) -> dict[str, Any]:
    return validate_specification(_json_object(text))


def fallback_specification(task: str) -> dict[str, Any]:
    """Build a deterministic spec when hosted planning is unavailable."""
    task = re.sub(r"\s+", " ", task).strip()
    calculator = bool(re.search(r"\b(calculator|calculater|calc)\b", task, re.IGNORECASE))
    folder_match = re.search(r"\b(?:folder|directory)\s+(?:named\s+|called\s+|ka\s+naam\s+)?[\"']?([A-Za-z][\w-]*)", task, re.IGNORECASE)
    project_dir = folder_match.group(1).lower() if folder_match and folder_match.group(1).lower() not in {"banao", "create", "make", "aur", "and"} else None
    if calculator:
        project_dir = project_dir or "calculator"
        prefix = project_dir + "/"
        paths = [prefix + path for path in ("main.py", "calculator_logic.py", "README.md", "tests/test_calculator_logic.py")]
        return validate_specification({
            "task_type": "calculator_development", "project_name": project_dir,
            "intended_outcome": task, "language": "Python",
            "framework": "tkinter GUI with safe AST expression evaluator",
            "features": ["basic arithmetic", "decimal and negative numbers", "parentheses and operator precedence", "clear, backspace, and input validation"],
            "expected_files": [{"path": path, "responsibility": "Calculator source, usage guide, or focused tests"} for path in paths],
            "dependencies": [], "run_commands": [f"python {prefix}main.py"],
            "test_commands": [f"python -m unittest discover -s {prefix}tests -v"],
            "acceptance_criteria": ["Arithmetic is evaluated without unrestricted eval", "Core arithmetic tests pass", "GUI starts when a display is available"],
            "assumptions": ["Use standard-library tkinter and no network-installed dependencies."],
            "definition_of_done": "Calculator source, documentation, and tests exist and core arithmetic tests pass.",
        })
    game = bool(re.search(r"\b(game|खेल)\b", task, re.IGNORECASE))
    named_files = list(dict.fromkeys(re.findall(
        r"(?i)(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_-]+\.(?:py|pyw|js|jsx|ts|tsx|html|css|json|md|txt|java|cpp|c|cs|go|rs|php|sql|yml|yaml|toml|sh|bat|ps1)",
        task,
    )))
    name_match = re.search(r"\b(?:called|named|name|titled)\s+([A-Za-z][\w-]*)", task, re.IGNORECASE)
    name = name_match.group(1).lower() if name_match else ("game" if game else "project")
    if game:
        expected = [
            {"path": "main.py", "responsibility": "Playable game entry point and interface"},
            {"path": "README.md", "responsibility": "Setup and run instructions"},
            {"path": "requirements.txt", "responsibility": "Runtime dependencies"},
            {"path": "tests/test_game_logic.py", "responsibility": "Automated tests for core game rules"},
        ]
        return validate_specification({
            "task_type": "game_development", "project_name": name,
            "intended_outcome": "A playable game with real interactive gameplay, not a placeholder.",
            "language": "Python", "framework": "Tkinter with procedural graphics",
            "features": ["game loop", "keyboard input", "game rules and state updates", "score or progression", "collision and game-over state", "restart behavior"],
            "expected_files": expected, "dependencies": [],
            "run_commands": ["python main.py"],
            "test_commands": ["python -m unittest discover -s tests -v"],
            "acceptance_criteria": ["Core gameplay rules are tested", "The game responds to player input", "The README documents setup and controls"],
            "assumptions": ["Use a small, self-contained 2D game with procedural visuals and Python's standard-library GUI unless the user specified a framework."],
            "definition_of_done": "Expected files exist, contain a playable game rather than placeholder output, and automated game logic tests pass.",
        })
    website = bool(re.search(r"\b(website|web site|web page|frontend|landing page)\b", task, re.IGNORECASE))
    api = bool(re.search(r"\b(api|backend|rest service)\b", task, re.IGNORECASE))
    cli = bool(re.search(r"\b(cli|command line|terminal app)\b", task, re.IGNORECASE))
    creation = bool(re.search(r"\b(build|create|make|implement|banao|bana do|develop)\b", task, re.IGNORECASE))
    if website or api or cli or creation:
        prefix = f"{project_dir}/" if project_dir else ""
        if website:
            task_type, language, framework = "website_development", "HTML, CSS, and JavaScript", "Vanilla browser APIs"
            paths = ("index.html", "styles.css", "script.js", "README.md", "tests/test_app.py")
        elif api:
            task_type, language, framework = "api_development", "Python", "Python standard-library HTTP server"
            paths = ("main.py", "README.md", "tests/test_api.py")
        elif cli:
            task_type, language, framework = "cli_development", "Python", "Python standard library"
            paths = ("main.py", "README.md", "tests/test_cli.py")
        else:
            task_type, language, framework = "software_development", "Use the repository's existing language", "Preserve existing conventions"
            paths = ("main.py", "README.md", "tests/test_main.py")
        return validate_specification({
            "task_type": task_type, "project_name": name,
            "intended_outcome": task, "language": language, "framework": framework,
            "features": [task],
            "expected_files": [{"path": prefix + path, "responsibility": "Application source, usage instructions, or focused tests"} for path in paths],
            "dependencies": [], "run_commands": [f"python {prefix}main.py"] if not website else [],
            "test_commands": [f"python -m unittest discover -s {prefix}tests -v"],
            "acceptance_criteria": ["The requested application behavior is implemented", "Relevant focused tests pass", "The README explains how to run the application"],
            "assumptions": ["Prefer locally available standard-library or browser capabilities; do not install packages during offline work."],
            "definition_of_done": "Application source, usage documentation, and focused tests exist and verification passes.",
        })
    return validate_specification({
        "task_type": "software_change", "project_name": name,
        "intended_outcome": task.strip(), "language": "Use the repository's existing language",
        "framework": "Preserve the existing project framework", "features": [task.strip()],
        "expected_files": [{"path": path, "responsibility": "Explicitly requested deliverable"} for path in named_files],
        "dependencies": [], "run_commands": [], "test_commands": [],
        "acceptance_criteria": ["Requested behavior is implemented", "Relevant checks pass"],
        "assumptions": ["Inspect the repository and preserve its conventions."],
        "definition_of_done": "The requested change is implemented and verified with relevant evidence.",
    })


def completeness_issues(workspace: str, specification: dict[str, Any]) -> list[str]:
    """Check required outputs and reject trivial/placeholder game projects."""
    issues: list[str] = []
    root = os.path.realpath(workspace)
    contents: list[tuple[str, str]] = []
    for item in specification.get("expected_files", []):
        path = item["path"]
        full = os.path.realpath(os.path.join(root, path))
        try:
            if os.path.commonpath([root, full]) != root:
                issues.append(f"Required path escapes workspace: {path}")
                continue
        except ValueError:
            issues.append(f"Required path escapes workspace: {path}")
            continue
        if not os.path.isfile(full):
            issues.append(f"Missing required file: {path}")
            continue
        try:
            text = Path(full).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            issues.append(f"Cannot read required file {path}: {exc}")
            continue
        if not text.strip():
            issues.append(f"Required file is empty: {path}")
            continue
        contents.append((path.lower(), text))

    if specification.get("task_type", "").lower() in {"game", "game_development"}:
        sources = [text for path, text in contents if path.endswith((".py", ".js", ".ts", ".html")) and not Path(path).name.startswith("test_")]
        source = "\n".join(sources)
        code = re.sub(r"(?m)^\s*#.*$", "", source).strip()
        if len(code) < 240 or re.search(r"(?is)^\s*(?:print\s*\(\s*)?[\"']hello\s+world[\"']\s*\)?\s*$", code):
            issues.append("Game implementation is trivial or placeholder code, not a playable game")
        if not re.search(r"(?i)\b(while|for|run|game_loop|event\.get|mainloop|update)\b", code):
            issues.append("Game source has no detectable gameplay/update loop")
        if not re.search(r"(?i)\b(key|keyboard|keydown|key_down|event|input|mouse|wasd|arrows)\b", code):
            issues.append("Game source has no detectable player input handling")
        if re.search(r"(?im)^\s*(?:pass|TODO|NotImplementedError)\s*$", code):
            issues.append("Game source still contains an unimplemented placeholder")
    return issues
