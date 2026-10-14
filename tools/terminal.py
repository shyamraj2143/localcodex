"""Workspace command execution with structured, factual process outcomes."""

import os
import locale
import subprocess
import time
import re


OFFLINE_NETWORK_COMMAND = re.compile(
    r"(?:\bpip(?:3)?\s+install\b|\b(?:uv|poetry)\s+(?:add|install|sync)\b|"
    r"\b(?:npm|pnpm|yarn)\s+(?:install|add)\b|\b(?:cargo|go)\s+(?:install|get)\b|"
    r"\b(?:curl|wget|Invoke-WebRequest|iwr)\b|\b(?:ollama)\s+pull\b|"
    r"\bgit\s+clone\b|\b(?:ping|nslookup|tracert|Test-NetConnection)\b|"
    r"https?://)", re.IGNORECASE,
)


def run_command(workspace, command, timeout=120, offline=False):
    workspace = os.path.realpath(os.path.abspath(workspace))
    command = str(command)
    started = time.monotonic()
    dangerous = [
        "format ",
        "diskpart",
        "shutdown",
        "rmdir /s",
        "del /s /q",
        "rm -rf",
        "sudo"
    ]

    command_lower = command.lower()
    if offline and OFFLINE_NETWORK_COMMAND.search(command):
        return {
            "success": False, "command": command, "cwd": workspace,
            "return_code": None, "stdout": "", "stderr": "",
            "duration_seconds": time.monotonic() - started,
            "timed_out": False, "blocked": True,
            "error": "Blocked network-dependent command in offline mode. Use an already installed local dependency/tool.",
        }
    for item in dangerous:
        if item in command_lower:
            return {
                "success": False, "command": command, "cwd": workspace,
                "return_code": None, "stdout": "", "stderr": "",
                "duration_seconds": time.monotonic() - started,
                "timed_out": False, "blocked": True,
                "error": f"Blocked dangerous command: {item}",
            }

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "success": result.returncode == 0,
            "command": command,
            "cwd": workspace,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": time.monotonic() - started,
            "timed_out": False,
            "blocked": False,
        }
    except subprocess.TimeoutExpired as error:
        def decode(value):
            if isinstance(value, bytes):
                return value.decode(locale.getpreferredencoding(False), errors="replace")
            return value or ""
        return {
            "success": False,
            "command": command,
            "cwd": workspace,
            "return_code": None,
            "stdout": decode(error.stdout),
            "stderr": decode(error.stderr),
            "duration_seconds": time.monotonic() - started,
            "timed_out": True,
            "blocked": False,
            "error": f"Command timed out after {timeout} seconds.",
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {
            "success": False,
            "command": command,
            "cwd": workspace,
            "return_code": None,
            "stdout": "",
            "stderr": str(error),
            "duration_seconds": time.monotonic() - started,
            "timed_out": False,
            "blocked": False,
            "error": str(error),
        }
