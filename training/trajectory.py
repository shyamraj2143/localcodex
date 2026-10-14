"""Compact, sanitized execution records for later curation and training export."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import os

from agent.orchestration import redact_secrets


def _safe(value):
    if isinstance(value, str):
        return redact_secrets(value)[:2400]
    if isinstance(value, list):
        return [_safe(item) for item in value[:100]]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in list(value.items())[:100]}
    return value


def build_trajectory_record(report: dict, workspace_id: str, memory_fingerprint: str | None = None) -> dict:
    commands = report.get("commands", [])
    memory_context = report.get("memory_context")
    memory_context = memory_context if isinstance(memory_context, dict) else {}
    return {
        "schema_version": 1,
        "workspace_id": workspace_id,
        "memory_fingerprint": memory_fingerprint,
        "task_id": report.get("task_id"),
        "user_request": redact_secrets(str(report.get("task", "")))[:2000],
        "workspace_context": redact_secrets(str(report.get("context_snapshot", "")))[:16000],
        "retrieved_paths": memory_context.get("retrieved_paths", [])[:30],
        "files_changed": report.get("changed_files", [])[:100],
        "source_changes": _safe(report.get("source_changes", {})),
        "final_report": redact_secrets(str(report.get("final_report", "")))[:8000],
        "tool_results": [{key: redact_secrets(str(row.get(key, "")))[:2400] for key in
                           ("command", "return_code", "success", "stdout", "stderr", "duration_seconds", "timed_out") if key in row}
                          for row in commands[-30:]],
        "verification_results": _safe(report.get("static_checks", [])[-40:]),
        "verification_status": report.get("verification_status", "unknown"),
        "failure_category": None if report.get("verification_status") == "PASS" else "incomplete_or_unverified",
        "recorded_at": report.get("finished_at"),
    }


def curate_trajectories(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate verified demonstrations from failed trajectories for diagnosis."""
    verified, diagnostics = [], []
    for row in records:
        if not isinstance(row, dict):
            continue
        if row.get("verification_status") != "PASS":
            diagnostics.append({**_safe(row), "failure_category": row.get("failure_category") or "incomplete_or_unverified"})
            continue
        context = redact_secrets(str(row.get("workspace_context", "")))[:16000]
        prompt = f"Task: {redact_secrets(str(row.get('user_request', '')))[:2000]}\n\nRelevant workspace context:\n{context}"
        answer_parts = [redact_secrets(str(row.get("final_report", "")))[:8000]]
        source_changes = row.get("source_changes")
        if isinstance(source_changes, dict):
            for path, content in list(source_changes.items())[:12]:
                answer_parts.append(f"\nVerified source change {path}:\n{redact_secrets(str(content))[:8000]}")
        response = "\n".join(part for part in answer_parts if part.strip())
        if not prompt.strip() or not response.strip():
            diagnostics.append({**_safe(row), "failure_category": "verified_record_missing_training_content"})
            continue
        verified.append({"prompt": prompt, "response": response,
                         "metadata": {"workspace_id": row.get("workspace_id"), "task_id": row.get("task_id"),
                                      "memory_fingerprint": row.get("memory_fingerprint"), "verification_status": "PASS"}})
    return verified, diagnostics


def write_trajectory_jsonl(path: str | Path, records: list[dict]) -> None:
    """Atomically replace a curated export; keep failed records labeled, never positive examples."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in records:
                safe = dict(row)
                if safe.get("verification_status") != "PASS":
                    safe["failure_category"] = safe.get("failure_category") or "incomplete_or_unverified"
                stream.write(json.dumps(safe, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
