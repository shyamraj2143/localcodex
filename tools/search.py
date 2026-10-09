"""Workspace text search for Local Codex."""

import os

from tools.files import IGNORED_DIRS, MAX_SEARCH_FILE_BYTES, safe_path


def search_files(workspace, query):
    """Return relative paths of text files containing a case-insensitive query."""
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "Search query must not be empty."}

    workspace = os.path.realpath(os.path.abspath(workspace))
    needle = query.casefold()
    results = []
    skipped_large = 0

    for root, dirs, filenames in os.walk(workspace):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)

        for filename in filenames:
            path = os.path.join(root, filename)
            try:
                if os.path.getsize(path) > MAX_SEARCH_FILE_BYTES:
                    skipped_large += 1
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        if needle in line.casefold():
                            results.append(os.path.relpath(path, workspace))
                            break
            except (OSError, PermissionError):
                continue

    unique_results = sorted(set(results))
    return {
        "success": True,
        "query": query,
        "matches": unique_results[:500],
        "count": len(unique_results),
        "truncated": len(unique_results) > 500,
        "skipped_large_files": skipped_large,
    }
