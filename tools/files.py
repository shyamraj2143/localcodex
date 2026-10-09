"""Workspace-scoped file tools used by Local Codex."""

import os

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vscode", ".pytest_cache", ".mypy_cache", "dist", "build",
    ".next", "coverage",
}
MAX_READ_CHARS = 100_000
MAX_SEARCH_FILE_BYTES = 2_000_000


def safe_path(workspace, path):
    """Resolve a file path and reject traversal and symlink escapes."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("A non-empty relative file path is required.")

    root = os.path.realpath(os.path.abspath(workspace))
    candidate = os.path.realpath(os.path.join(root, path))

    try:
        common = os.path.commonpath([root, candidate])
    except ValueError as error:
        raise ValueError("Path must stay inside the workspace.") from error

    if os.path.normcase(common) != os.path.normcase(root):
        raise ValueError("Access outside project is not allowed.")

    return candidate


def list_files(workspace):
    """List workspace files while skipping dependency/build/cache folders."""
    workspace = os.path.realpath(os.path.abspath(workspace))
    files = []

    for root, dirs, filenames in os.walk(workspace):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for filename in sorted(filenames):
            full_path = os.path.join(root, filename)
            files.append(os.path.relpath(full_path, workspace))

    return sorted(files)


def read_file(workspace, path):
    full_path = safe_path(workspace, path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(path)

    with open(full_path, "r", encoding="utf-8", errors="replace") as file:
        content = file.read(MAX_READ_CHARS + 1)

    if len(content) > MAX_READ_CHARS:
        content = (
            content[:MAX_READ_CHARS]
            + "\n\n...[file truncated at 100,000 characters; use search_files "
              "or inspect a smaller relevant file]"
        )
    return content


def write_file(workspace, path, content):
    if not isinstance(content, str):
        raise TypeError("File content must be a string.")

    full_path = safe_path(workspace, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8", newline="") as file:
        file.write(content)

    return {
        "success": True,
        "message": f"Created/updated {path}",
        "path": os.path.relpath(full_path, os.path.realpath(workspace)),
        "bytes_written": len(content.encode("utf-8")),
    }


def edit_file(workspace, path, old_text, new_text):
    """Replace exactly one unique text block in an existing file."""
    full_path = safe_path(workspace, path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(path)
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text must be a non-empty string.")
    if not isinstance(new_text, str):
        raise TypeError("new_text must be a string.")

    with open(full_path, "r", encoding="utf-8", errors="replace") as file:
        content = file.read()

    occurrences = content.count(old_text)
    if occurrences == 0:
        return {
            "success": False,
            "error": f"Text to replace was not found in {path}.",
        }
    if occurrences > 1:
        return {
            "success": False,
            "error": (
                f"old_text occurs {occurrences} times in {path}; "
                "provide a larger unique block."
            ),
        }

    new_content = content.replace(old_text, new_text, 1)
    with open(full_path, "w", encoding="utf-8", newline="") as file:
        file.write(new_content)

    return {
        "success": True,
        "message": f"Edited {path}",
        "path": os.path.relpath(full_path, os.path.realpath(workspace)),
    }
