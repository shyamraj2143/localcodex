"""Safe, offline dataset loading, validation, preparation, and splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|token|password|secret)\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{8,})"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
SECRET_ASSIGNMENT = re.compile(r"(?i)(api[_-]?key|access[_-]?token|token|password|secret)\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{8,})")


class DatasetError(ValueError):
    """Raised when a dataset cannot be parsed safely."""


def _jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise DatasetError(f"Line {line_no} must contain a JSON object")
            yield value


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"Dataset file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return list(_jsonl_records(source))
    if suffix == ".json":
        try:
            value = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"Could not parse JSON dataset: {exc}") from exc
        if isinstance(value, dict):
            value = value.get("data", [value])
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise DatasetError("JSON dataset must be an array of objects or an object with a data array")
        return value
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".parquet":
        try:
            import pandas  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DatasetError("Parquet input requires the optional pandas and pyarrow dependencies") from exc
        return pandas.read_parquet(source).to_dict(orient="records")
    raise DatasetError("Supported formats: .json, .jsonl, .ndjson, .csv, .parquet")


def _messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list):
        result = []
        for item in messages:
            if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"} or not isinstance(item.get("content"), str):
                raise DatasetError("messages must contain {role, content} entries for system/user/assistant")
            if item["content"].strip():
                result.append({"role": item["role"], "content": item["content"].strip()})
        return result
    prompt = row.get("prompt", row.get("instruction", row.get("task", row.get("input"))))
    answer = row.get("response", row.get("output", row.get("completion", row.get("solution"))))
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise DatasetError("Record requires messages[] or string prompt/instruction/task and response/output fields")
    return [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer.strip()}]


def _redact(text: str) -> tuple[str, bool]:
    redacted = text
    found = False
    for pattern in SECRET_PATTERNS[1:]:
        redacted, count = pattern.subn("[REDACTED_SECRET]", redacted)
        found |= count > 0
    redacted, count = SECRET_ASSIGNMENT.subn(lambda match: f"{match.group(1)}=[REDACTED_SECRET]", redacted)
    found |= count > 0
    return redacted, found


def normalize_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"input": len(rows), "accepted": 0, "duplicates": 0, "rejected": 0, "secret_redactions": 0, "rejection_reasons": {}}
    for index, row in enumerate(rows):
        try:
            messages = _messages(row)
            if len(messages) < 2 or not any(m["role"] == "user" and m["content"] for m in messages) or not any(m["role"] == "assistant" and m["content"] for m in messages):
                raise DatasetError("record must include non-empty user and assistant messages")
            sanitized = []
            redacted = False
            for message in messages:
                content, changed = _redact(message["content"])
                sanitized.append({"role": message["role"], "content": content})
                redacted |= changed
            material = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).casefold()
            digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if digest in seen:
                stats["duplicates"] += 1
                continue
            seen.add(digest)
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"source_metadata": str(metadata)}
            record = {"id": str(row.get("id") or digest[:16]), "messages": sanitized, "metadata": metadata, "content_sha256": digest}
            if redacted:
                record["metadata"] = {**metadata, "secrets_redacted": True}
                stats["secret_redactions"] += 1
            accepted.append(record)
        except DatasetError as exc:
            stats["rejected"] += 1
            reason = str(exc)
            stats["rejection_reasons"][reason] = stats["rejection_reasons"].get(reason, 0) + 1
    stats["accepted"] = len(accepted)
    return accepted, stats


def split_records(records: list[dict[str, Any]], seed: int = 42, train: float = 0.8, validation: float = 0.1) -> dict[str, list[dict[str, Any]]]:
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise ValueError("Split ratios must satisfy 0 < train < 1, validation >= 0, and train + validation < 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        # Duplicate hashes are kept in one partition even when supplied with different IDs.
        key = str(row.get("content_sha256") or hashlib.sha256(json.dumps(row.get("messages", row), sort_keys=True).encode()).hexdigest())
        groups.setdefault(key, []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    train_count = int(len(keys) * train)
    validation_count = int(len(keys) * validation)
    test_count = len(keys) - train_count - validation_count
    if keys and train_count == 0:
        train_count, test_count = 1, max(0, test_count - 1)
    if len(keys) >= 3 and validation_count == 0:
        validation_count, train_count = 1, train_count - 1
    if len(keys) >= 2 and test_count == 0:
        test_count, train_count = 1, train_count - 1
    train_end = train_count
    validation_end = train_end + validation_count
    partitions = {"train": keys[:train_end], "validation": keys[train_end:validation_end], "test": keys[validation_end:]}
    result = {name: [row for key in part for row in groups[key]] for name, part in partitions.items()}
    return result


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
