"""Workspace command execution with structured, factual process outcomes."""

import os
import locale
import signal
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
        timeout = max(0.1, float(timeout))
        process_options = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": locale.getpreferredencoding(False),
            "errors": "replace",
        }
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=workspace,
            **process_options,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            if os.name == "nt":
                # `shell=True` starts cmd.exe; terminate its whole process tree so
                # child commands cannot outlive the agent's timeout.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    process.kill()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            return {
                "success": False,
                "command": command,
                "cwd": workspace,
                "return_code": process.returncode,
                "stdout": stdout or _decode_timeout_output(error.stdout),
                "stderr": stderr or _decode_timeout_output(error.stderr),
                "duration_seconds": time.monotonic() - started,
                "timed_out": True,
                "blocked": False,
                "error": f"Command timed out after {timeout:g} seconds and was terminated.",
            }
        return {
            "success": process.returncode == 0,
            "command": command,
            "cwd": workspace,
            "return_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_seconds": time.monotonic() - started,
            "timed_out": False,
            "blocked": False,
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


def _decode_timeout_output(value):
    if isinstance(value, bytes):
        return value.decode(locale.getpreferredencoding(False), errors="replace")
    return value or ""
