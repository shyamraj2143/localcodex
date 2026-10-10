"""Agentic coding loop for Local Codex.

Unlike the previous planner + parallel-worker pipeline, this loop keeps one
coding agent in control of inspect -> edit -> run -> diagnose -> verify. This
is more reliable for small local Ollama models and mirrors the core workflow
of an interactive coding agent.
"""

import json
import os
import re
import sys
import uuid
from types import SimpleNamespace

from agent.groq import GroqClient
from agent.nvidia import NvidiaClient
from agent.ollama import OllamaClient
from agent.verification import verify_changed_file


MAX_TOOL_RESULT_CHARS = 16000
MAX_SESSION_TURNS = 8


def _iter_json_objects(content):
    """Yield JSON objects embedded in plain text, including multiple fenced blocks."""
    if not isinstance(content, str) or not content.strip():
        return

    decoder = json.JSONDecoder()
    seen = set()
    for match in re.finditer(r"\{", content):
        start = match.start()
        try:
            parsed, end = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        fingerprint = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield parsed


def _extract_json_object(content):
    """Parse the first JSON object from plain text or a fenced code block."""
    return next(_iter_json_objects(content), None)


def recover_text_tool_calls(content, available_tools):
    """Convert JSON-shaped tool calls embedded anywhere in text into executable calls."""
    recovered = []
    seen_calls = set()
    for data in _iter_json_objects(content):
        calls = data.get("tool_calls")
        if not isinstance(calls, list):
            calls = [data]

        for item in calls:
            if not isinstance(item, dict):
                continue

            function_data = item.get("function")
            if not isinstance(function_data, dict):
                function_data = {}

            name = item.get("name") or function_data.get("name")
            arguments = item.get("arguments", function_data.get("arguments", {}))

            if not isinstance(name, str) or name not in available_tools:
                continue

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    continue

            if arguments == []:
                arguments = {}
            if not isinstance(arguments, dict):
                continue

            signature = (
                name,
                json.dumps(arguments, sort_keys=True, ensure_ascii=False),
            )
            if signature in seen_calls:
                continue
            seen_calls.add(signature)
            recovered.append(
                SimpleNamespace(
                    id="call_" + uuid.uuid4().hex,
                    type="function",
                    function=SimpleNamespace(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
    return recovered

def _is_tool_call_shaped_json(content):
    data = _extract_json_object(content)
    return isinstance(data, dict) and (
        "name" in data or "function" in data or "tool_calls" in data
    )


def _is_tool_result_json(content):
    """Detect a tool result that a small model mistakenly echoes as its answer."""
    data = _extract_json_object(content)
    if not isinstance(data, dict) or not isinstance(data.get("success"), bool):
        return False
    result_fields = {
        "message", "path", "absolute_path", "bytes_written",
        "return_code", "stdout", "stderr", "error", "output",
    }
    return bool(result_fields.intersection(data))


def _is_meaningful_verification_command(command):
    """Reject unrelated successful commands (for example, pip --version) as code verification."""
    if not isinstance(command, str):
        return False
    command = command.lower()
    patterns = (
        r"\bpytest\b", r"\bunittest\b", r"\bcompileall\b",
        r"\bpy_compile\b", r"\bnode\s+--check\b",
        r"\bnpm\s+(?:test|run\s+(?:test|build|lint|typecheck))\b",
        r"\bnpx\s+(?:tsc|eslint|vitest|jest)\b",
        r"\btsc\s+--noemit\b", r"\beslint\b", r"\bprettier\s+--check\b",
        r"\bruff\s+check\b", r"\bmypy\b", r"\bgo\s+test\b",
        r"\bcargo\s+test\b", r"\bmvn\s+test\b", r"\bgradle\s+test\b",
        r"\bphp\s+-l\b", r"\bdotnet\s+test\b",
    )
    return any(re.search(pattern, command) for pattern in patterns)


def _expected_artifact_paths(task):
    """Infer explicit file/folder deliverables so premature 'done' is caught."""
    if not isinstance(task, str):
        return []

    folders = re.findall(
        r"\\bfolder\\s+(?:named|called|name|ka\\s+naam)\\s+['\" ]?([A-Za-z0-9_-]+(?:\\.[A-Za-z0-9_-]+)*)",
        task,
        flags=re.IGNORECASE,
    )
    # Hinglish commonly puts the folder name after 'folder' and before 'banao'.
    folders.extend(re.findall(
        r"\\bfolder\\s+['\" ]?([A-Za-z0-9_-]+)['\" ]?\\s+(?:banao|banaye|bana\\s+do|create\\s+karo)",
        task,
        flags=re.IGNORECASE,
    ))
    # Also understand common Hinglish requests such as "folder banao gui".
    folders.extend(re.findall(
        r"\bfolder\s+(?:banao|banaye|bana\s+do|create\s+karo)\s+([A-Za-z0-9_-]+)",
        task,
        flags=re.IGNORECASE,
    ))
    folders.extend(re.findall(
        r"\b([A-Za-z0-9_-]+)\s+folder\s+(?:banao|banaye|bana\s+do)",
        task,
        flags=re.IGNORECASE,
    ))
    stop_words = {"aur", "and", "or", "usme", "isme", "then", "with", "code"}
    folders = list(dict.fromkeys(
        folder for folder in folders if folder.lower() not in stop_words
    ))
    # Capture explicitly mentioned source/config files even when the user phrases
    # the request in Hindi/Hinglish ("main.py mein code likho").
    file_paths = re.findall(
        r"((?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_-]+\."
        r"(?:py|pyw|js|jsx|ts|tsx|html|css|json|md|txt|java|cpp|c|cs|go|rs|php|sql|"
        r"yml|yaml|toml|sh|bat|ps1|env))\b",
        task,
        flags=re.IGNORECASE,
    )

    expected = list(folders)
    inside_folder = bool(
        re.search(
            r"\binside\s+(?:it|that folder|the folder)\b|"
            r"\b(folder|directory)\s+ke\s+andar\b|"
            r"\busme\b",
            task,
            re.IGNORECASE,
        )
    )
    for file_path in file_paths:
        normalized = file_path.replace("\\", "/")
        if inside_folder and folders and "/" not in normalized:
            normalized = folders[-1] + "/" + normalized
        if normalized not in expected:
            expected.append(normalized)

    # Only return relative paths; all actual operations remain workspace-scoped.
    return [path for path in expected if path and not os.path.isabs(path)]


class MultiModelAgent:
    """A single-agent, tool-driven coding loop with persistent short history."""

    def __init__(self, config, tools, tool_schemas, workspace, status=None):
        self.config = config
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.workspace = workspace
        self.status_callback = status
        self.models = config.get("models", {})
        self.max_workers = max(1, min(int(config.get("max_parallel_workers", 1)), 4))
        self.max_steps = max(1, min(int(config.get("max_agent_steps", 20)), 40))
        self.session_history = []
        self.last_task_report = {}

    def _format_final_report(self, summary):
        """Append a factual execution report based only on recorded tool results."""
        report = self.last_task_report or {}
        changed = list(dict.fromkeys(report.get("changed_files", [])))
        commands = report.get("commands", [])
        checks = report.get("static_checks", [])

        lines = [str(summary or "").strip(), "", "Execution report:"]
        lines.append("Files changed: " + (", ".join(changed) if changed else "none recorded"))

        latest_checks = {}
        for item in checks:
            latest_checks[item.get("path", "unknown")] = item
        if latest_checks:
            lines.append("Automatic file checks:")
            for path, item in latest_checks.items():
                state = "PASS" if item.get("success") else "FAIL"
                details = "; ".join(item.get("checks", []) + item.get("errors", []))
                lines.append(f"- {state} {path}" + (f" — {details}" if details else ""))

        if commands:
            lines.append("Commands executed:")
            for item in commands[-8:]:
                state = "PASS" if item.get("success") and item.get("return_code") == 0 else "FAIL"
                lines.append(f"- {state} [{item.get('return_code', 'unknown')}] {item.get('command', '')}")

        meaningful = [item for item in commands if item.get("meaningful_check")]
        latest_meaningful_pass = bool(
            meaningful and meaningful[-1].get("success")
            and meaningful[-1].get("return_code") == 0
        )
        static_pass = all(item.get("success") for item in latest_checks.values())
        if changed and latest_meaningful_pass and static_pass:
            lines.append("Verification status: PASS — automatic file checks and the latest relevant test/check command passed.")
        elif changed:
            reasons = []
            if not latest_meaningful_pass:
                reasons.append("no successful relevant test/syntax/build command was recorded after the changes")
            if not static_pass:
                reasons.append("one or more latest automatic file checks failed")
            lines.append("Verification status: PARTIAL — " + "; ".join(reasons) + ".")
        else:
            lines.append("Verification status: no file modifications were recorded; review the task summary above.")
        return "\n".join(line for line in lines if line is not None)

    def _missing_expected_artifacts(self, task_context):
        """Return explicitly requested relative paths that do not exist yet."""
        missing = []
        root = os.path.realpath(self.workspace)
        for artifact in _expected_artifact_paths(task_context):
            full_path = os.path.realpath(os.path.join(root, artifact))
            try:
                if os.path.commonpath([root, full_path]) != root:
                    continue
            except ValueError:
                continue
            if not os.path.exists(full_path):
                missing.append(artifact)
        return missing

    def _generate_missing_files_locally(self, client, task_context, expected_paths):
        """Escape a stuck tool loop by asking the local model for file contents without tools."""
        file_paths = [
            path for path in expected_paths
            if os.path.splitext(path)[1].lower() in {
                ".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".html", ".css",
                ".json", ".md", ".txt", ".java", ".cpp", ".c", ".cs", ".go",
                ".rs", ".php", ".sql", ".yml", ".yaml", ".toml", ".sh",
            }
        ]
        if not file_paths:
            return False

        last_error = ""
        for attempt in range(2):
            self.show_status("Local model stalled; generating required files directly")
            prompt = (
                "You are recovering a stalled local coding agent. Return ONLY one valid JSON object "
                "with this exact shape: {\"files\": {\"relative/path.ext\": \"complete file content\", ...}}. "
                "No Markdown fences, no commentary, and no tool-call JSON.\n"
                "Complete the user's entire task in the listed files. Include complete source code and "
                "tests when requested. Do not return folder names as files.\n"
                f"Workspace root: {self.workspace}\n"
                f"Required file paths (use these exact keys): {json.dumps(file_paths, ensure_ascii=False)}\n"
                f"Original task:\n{task_context}\n"
                + (f"Previous verification failure to fix:\n{last_error}\n" if last_error else "")
            )
            try:
                response = client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Generate complete source files as a JSON files map. "
                                "The response is consumed by a program, so valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tools=None,
                )
                content = getattr(response.choices[0].message, "content", "") or ""
                payload = _extract_json_object(content)
                generated = payload.get("files") if isinstance(payload, dict) else None
                if not isinstance(generated, dict):
                    last_error = "Model did not return a JSON object with a files map."
                    continue

                allowed = {os.path.normcase(os.path.normpath(path)): path for path in file_paths}
                wrote_any = False
                for path, file_content in generated.items():
                    if not isinstance(path, str) or not isinstance(file_content, str):
                        continue
                    normalized = os.path.normcase(os.path.normpath(path.replace("\\", os.sep)))
                    canonical = allowed.get(normalized)
                    if not canonical:
                        continue
                    result = self.execute_tool("write_file", {
                        "path": canonical,
                        "content": file_content,
                    })
                    if not isinstance(result, dict) or result.get("success") is not True:
                        last_error = f"Could not write {canonical}: {result}"
                        continue
                    wrote_any = True
                    if canonical not in self.last_task_report["changed_files"]:
                        self.last_task_report["changed_files"].append(canonical)
                    check = verify_changed_file(self.workspace, canonical)
                    self.last_task_report["static_checks"].append(check)

                missing = self._missing_expected_artifacts(task_context)
                if missing:
                    last_error = "Still missing requested artifacts: " + ", ".join(missing)
                    if not wrote_any:
                        continue

                python_files = [path for path in file_paths if path.lower().endswith((".py", ".pyw"))]
                test_files = [path for path in python_files if os.path.basename(path).lower().startswith("test_")]
                if test_files:
                    test_dir = os.path.dirname(test_files[0]).replace("\\", "/") or "."
                    command = f'"{sys.executable}" -m unittest discover -s "{test_dir}" -v'
                elif python_files:
                    targets = " ".join(f'"{path}"' for path in python_files)
                    command = f'"{sys.executable}" -m py_compile {targets}'
                elif any(path.lower().endswith((".js", ".mjs", ".cjs")) for path in file_paths):
                    command = "node --check " + " ".join(
                        f'"{path}"' for path in file_paths if path.lower().endswith((".js", ".mjs", ".cjs"))
                    )
                else:
                    command = ""

                if command:
                    result = self.execute_tool("run_command", {"command": command})
                    record = {
                        "command": command,
                        "return_code": result.get("return_code") if isinstance(result, dict) else None,
                        "success": isinstance(result, dict) and result.get("success") is True,
                        "meaningful_check": _is_meaningful_verification_command(command),
                    }
                    self.last_task_report["commands"].append(record)
                    if record["success"] and record["return_code"] == 0:
                        return not self._missing_expected_artifacts(task_context)
                    last_error = (
                        f"Command failed (return code {record['return_code']}): {command}\n"
                        + str(result.get("stdout", "") if isinstance(result, dict) else result)
                        + "\n"
                        + str(result.get("stderr", "") if isinstance(result, dict) else "")
                    )
                    continue

                return not self._missing_expected_artifacts(task_context)
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"

        return False

    def show_status(self, message):
        if self.status_callback:
            self.status_callback(message)

    def get_client(self, role="coder"):
        provider = str(self.config.get("provider", "groq")).strip().lower()
        common = {
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 8192),
        }

        if provider in {"ollama", "local"}:
            return OllamaClient(
                model=self.config.get("local_model", "qwen2.5-coder:1.5b"),
                base_url=self.config.get(
                    "ollama_base_url", "http://localhost:11434/v1"
                ),
                force_json_tools=self.config.get("ollama_force_json_tools", True),
                **common,
            )

        if provider in {"nvidia", "nvidia_nim"}:
            retry = self.config.get("retry", {})
            return NvidiaClient(
                model=self.config.get(
                    "nvidia_model", "poolside/laguna-xs-2.1"
                ),
                base_url=self.config.get(
                    "nvidia_base_url", "https://integrate.api.nvidia.com/v1"
                ),
                **common,
                max_attempts=retry.get("max_attempts", 3),
                base_delay=retry.get("base_delay", 1.5),
            )

        if provider != "groq":
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'nvidia', 'groq', or 'ollama'."
            )

        model = self.models.get(role) or self.models.get("coder")
        if not model:
            raise KeyError(f"No model configured for role: {role}")

        retry = self.config.get("retry", {})
        return GroqClient(
            model=model,
            **common,
            max_attempts=retry.get("max_attempts", 3),
            base_delay=retry.get("base_delay", 1.5),
        )

    def get_project_snapshot(self):
        try:
            result = self.tools["list_files"](self.workspace)
            if isinstance(result, list):
                result = result[:500]
            return json.dumps(result, ensure_ascii=False)[:12000]
        except Exception as error:
            return f"Unable to scan project: {error}"

    def classify_task(self, task):
        text = task.lower()
        if any(word in text for word in (
            "build", "create", "implement", "application", "website",
            "backend", "frontend", "database", "authentication", "refactor",
            "deploy", "integrate",
        )):
            return "COMPLEX"
        if any(word in text for word in (
            "fix", "debug", "modify", "edit", "add", "feature", "bug", "error",
            "api", "component", "function",
        )):
            return "MEDIUM"
        return "SIMPLE"

    def execute_tool(self, name, arguments):
        if name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {name}"}
        if not isinstance(arguments, dict):
            return {"success": False, "error": "Tool arguments must be a JSON object."}

        try:
            result = self.tools[name](self.workspace, **arguments)
        except Exception as error:
            return {"success": False, "error": f"{type(error).__name__}: {error}"}

        # Bound tool output so a large repository or command output cannot flood context.
        try:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            serialized = str(result)

        if len(serialized) > MAX_TOOL_RESULT_CHARS:
            serialized = serialized[:MAX_TOOL_RESULT_CHARS] + "\n...[tool output truncated]"
            return {"success": True, "truncated": True, "output": serialized}
        return result

    def _status_for_tool(self, name, arguments):
        path = arguments.get("path", "") if isinstance(arguments, dict) else ""
        query = arguments.get("query", "") if isinstance(arguments, dict) else ""
        if name == "list_files":
            return "Inspecting workspace"
        if name == "read_file":
            return f"Reading {path}"
        if name == "write_file":
            return f"Writing {path}"
        if name == "edit_file":
            return f"Editing {path}"
        if name == "search_files":
            return f"Searching {query}"
        if name == "run_command":
            return "Running command"
        if name == "create_folder":
            return f"Creating folder {path}"
        return f"Running tool {name}"

    def _tool_message(self, tool_call):
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = {}

        if not isinstance(arguments, dict):
            result = {"success": False, "error": "Arguments must be a JSON object."}
        else:
            self.show_status(self._status_for_tool(name, arguments))
            result = self.execute_tool(name, arguments)

        if isinstance(result, dict) and result.get("success") is False:
            self.show_status("Command failed; analyzing error")

        try:
            output = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            output = json.dumps({"result": str(result)}, ensure_ascii=False)

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output[:MAX_TOOL_RESULT_CHARS],
        }

    def _assistant_tool_call_message(self, tool_calls):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }

    def integrate(self, task, reports=None):
        """Run the actual coding task until the model finishes or hits the step cap."""
        self.show_status("Thinking")
        client = self.get_client("coder")
        self.last_task_report = {
            "task": task,
            "changed_files": [],
            "commands": [],
            "static_checks": [],
        }

        tools_description = ", ".join(sorted(self.tools))
        provider = str(self.config.get("provider", "groq")).lower()
        local_json_mode = (
            provider in {"ollama", "local"}
            and bool(self.config.get("ollama_force_json_tools", True))
        )
        if provider in {"ollama", "local"}:
            # Small local models perform better with a compact, non-duplicated prompt.
            # The full tool schemas are supplied by OllamaClient's JSON protocol.
            system_prompt = f"""
You are Local Codex, an autonomous coding agent.
Workspace root: {self.workspace}
Tools available: {tools_description}

Understand the user's whole request in English, Hindi, or Hinglish, not just its first sentence. Preserve every explicit
requirement and requested deliverable. Infer sensible defaults for minor details.
Inspect the existing project before editing. Make real changes in workspace files; do not
just explain or print code. Work in small steps, one tool action per response.
After implementation, run an appropriate test or syntax check, inspect the result, fix
errors, and check again. Never claim a file or test succeeded without tool evidence.
Do not repeat successful actions. Keep all file operations inside the workspace.
If the task is genuinely ambiguous in a way that changes the result, ask one short question;
otherwise proceed with a reasonable implementation.
"""
        else:
            system_prompt = f"""
You are Local Codex, an autonomous software engineering agent working in:
{self.workspace}

Your job is to actually complete the user's coding task, not just explain how.
Available tools: {tools_description}

MANDATORY WORKFLOW:
1. Inspect the workspace and relevant files before changing code.
2. Make changes using the file tools; do not merely print code for the user to copy.
3. Use run_command to run relevant tests, linters, or the program when practical.
4. Read error output, diagnose the cause, edit the code, and run the check again.
5. Verify important outputs by reading the created/changed files or listing the result.
6. Continue using tools until the task is implemented and checked.
7. Never claim a file was created, a test passed, or a bug was fixed unless tool output verifies it.
8. Prefer small, focused changes. Preserve existing project conventions and unrelated work.
9. Ask a short question only when an essential requirement cannot be inferred safely.
10. Do not expose chain-of-thought. Give concise progress updates through tool use and a final summary.
11. For ordinary underspecified coding requests (for example, "make this GUI better"), infer sensible defaults, inspect the project, and implement them instead of asking the user to restate an obvious goal.
12. Treat the entire user message beginning with "Task:" as one complete task. Preserve all requirements, even if the prompt contains multiple paragraphs.
13. Before finishing, verify requested files exist and run a syntax check or relevant tests when practical. If a check fails, fix it and rerun the check.

TOOL CALL FORMAT:
- Prefer native function tool calls.
- If native calls are unavailable, call tools using one JSON object with keys "name" and "arguments".
- Use exact registered tool names and provide arguments as a JSON object.
- Never print a tool-call JSON object as your final answer.
- Tool results are real execution results. Use them to decide the next action.

Workspace root is the current directory. Keep file operations inside it.
"""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.session_history[-(MAX_SESSION_TURNS * 2):])
        messages.append({
            "role": "user",
            "content": (
                f"Task: {task}\n\n"
                "Work autonomously. Inspect, implement, run checks, fix failures, and verify."
            ),
        })

        malformed_tool_retries = 0
        completion_retries = 0
        empty_response_retries = 0
        disable_tools_next = False
        successful_mutation_calls = set()
        executed_tool_calls = 0
        mutation_needs_verification = False
        last_action_signature = None
        repeated_action_count = 0
        local_fallback_attempted = False
        recent_user_tasks = [
            item.get("content", "")
            for item in self.session_history
            if item.get("role") == "user"
        ][-2:]
        task_context = "\n".join(recent_user_tasks + [task])

        for step in range(self.max_steps):
            self.show_status(f"Thinking · step {step + 1}/{self.max_steps}")
            request_tools = None if disable_tools_next else self.tool_schemas
            disable_tools_next = False
            response = client.chat(messages, tools=request_tools)
            message = response.choices[0].message
            native_calls = getattr(message, "tool_calls", None) or []
            tool_calls = list(native_calls)
            recovered = False

            if not tool_calls:
                tool_calls = recover_text_tool_calls(message.content, self.tools)
                recovered = bool(tool_calls)

            if not tool_calls:
                raw_content = getattr(message, "content", None)
                final_text = (raw_content or "").strip()

                # Small local models occasionally return an empty assistant message
                # when tool schemas are included. Retry with a shorter instruction and
                # no schemas once, allowing the model to emit a recoverable JSON call.
                if not final_text and empty_response_retries < 3:
                    empty_response_retries += 1
                    if raw_content is not None:
                        messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. Continue the coding task. "
                            "Do not return an empty response. Choose the next action and "
                            "use exactly one JSON object like "
                            '{"name":"list_files","arguments":{}} '
                            "with a registered tool name and an arguments object, or provide "
                            "a concise final answer only after the task is complete."
                        ),
                    })
                    disable_tools_next = True
                    continue

                if not final_text:
                    final_text = (
                        "The local model returned an empty response after 3 retries. "
                        "Check that Ollama is running and the selected model is installed "
                        "with 'ollama list'. Very small models may struggle with multi-step "
                        "coding; try a stronger local coding model if memory permits."
                    )

                if (
                    _is_tool_call_shaped_json(final_text)
                    and malformed_tool_retries < 2
                ):
                    malformed_tool_retries += 1
                    messages.append({"role": "assistant", "content": final_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "That looks like a tool call but was not executable. "
                            f"Use only these exact registered tools: {tools_description}. "
                            "Return a native function call or JSON with keys name and arguments. "
                            "Arguments must be an object. Continue the task; do not stop."
                        ),
                    })
                    continue

                missing_artifacts = []
                for artifact in _expected_artifact_paths(task_context):
                    full_path = os.path.realpath(os.path.join(self.workspace, artifact))
                    workspace_root = os.path.realpath(self.workspace)
                    try:
                        inside_workspace = (
                            os.path.commonpath([workspace_root, full_path])
                            == workspace_root
                        )
                    except ValueError:
                        inside_workspace = False
                    if inside_workspace and not os.path.exists(full_path):
                        missing_artifacts.append(artifact)

                echoed_tool_result = _is_tool_result_json(final_text)
                no_local_work = local_json_mode and executed_tool_calls == 0
                unverified_changes = local_json_mode and mutation_needs_verification
                if (
                    (echoed_tool_result or missing_artifacts or no_local_work or unverified_changes)
                    and completion_retries < 4
                ):
                    completion_retries += 1
                    messages.append({"role": "assistant", "content": final_text})

                    reasons = []
                    if echoed_tool_result:
                        reasons.append(
                            "Your last message is a tool-result JSON object, not a user-facing final answer."
                        )
                    if missing_artifacts:
                        reasons.append(
                            "Required deliverables still missing: "
                            + ", ".join(missing_artifacts)
                            + "."
                        )
                    if no_local_work:
                        reasons.append(
                            "You have not executed any workspace tool yet. Inspect the project and do the requested work."
                        )
                    if unverified_changes:
                        reasons.append(
                            "You changed a source file but have not verified it with a relevant run_command check. Run the check, fix failures, and rerun it."
                        )

                    messages.append({
                        "role": "user",
                        "content": (
                            "The task is NOT complete. "
                            + " ".join(reasons)
                            + " Continue working now using the registered tools. "
                            "Create any missing files/folders, put the actual implementation in the files, "
                            "then verify them with read_file or run_command. Do not merely repeat tool output "
                            "or say 'Task completed'. Once everything exists and is verified, give a concise summary."
                        ),
                    })
                    continue

                if echoed_tool_result or missing_artifacts or no_local_work or unverified_changes:
                    details = []
                    if echoed_tool_result:
                        details.append("the model repeatedly echoed tool-result JSON")
                    if missing_artifacts:
                        details.append(
                            "missing deliverables: " + ", ".join(missing_artifacts)
                        )
                    if no_local_work:
                        details.append("no workspace tools were executed")
                    if unverified_changes:
                        details.append("source changes were not verified with a successful command")
                    final_text = (
                        "Local Codex could not verify task completion because "
                        + "; ".join(details)
                        + ". The model may be too small for this task. "
                        "Try Groq or a stronger local coding model, then ask it to continue."
                    )

                self.session_history.extend([
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": final_text},
                ])
                self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                return self._format_final_report(final_text)

            # In local JSON-tool mode, keep the transcript as ordinary user/assistant
            # messages. Synthetic assistant tool_calls + role=tool messages are often
            # mishandled by small Ollama models because no native tool schema was sent.
            if local_json_mode and recovered:
                messages.append({
                    "role": "assistant",
                    "content": getattr(message, "content", None) or "",
                })
            elif recovered:
                messages.append(self._assistant_tool_call_message(tool_calls))
            else:
                messages.append(message)

            # Execute one-action-at-a-time tool requests and give local models readable
            # results instead of a synthetic tool-call protocol they may echo forever.
            duplicate_mutation = False
            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                signature = (
                    name,
                    json.dumps(arguments, sort_keys=True, ensure_ascii=False),
                )
                if signature == last_action_signature:
                    repeated_action_count += 1
                else:
                    last_action_signature = signature
                    repeated_action_count = 1

                if name in {"create_folder", "write_file", "edit_file"} and signature in successful_mutation_calls:
                    tool_result = {
                        "success": False,
                        "error": (
                            "Duplicate mutation blocked: this exact action already "
                            "succeeded. Choose the next missing step."
                        ),
                    }
                    duplicate_mutation = True
                    tool_message = None
                else:
                    tool_message = self._tool_message(tool_call)
                    try:
                        tool_result = json.loads(tool_message["content"])
                    except (json.JSONDecodeError, TypeError):
                        tool_result = {"success": False, "error": tool_message["content"]}

                    if (
                        name in {"create_folder", "write_file", "edit_file"}
                        and isinstance(tool_result, dict)
                        and tool_result.get("success") is True
                    ):
                        successful_mutation_calls.add(signature)

                executed_tool_calls += 1
                if (
                    name in {"write_file", "edit_file"}
                    and isinstance(tool_result, dict)
                    and tool_result.get("success") is True
                ):
                    path = arguments.get("path", "")
                    if path:
                        if path not in self.last_task_report["changed_files"]:
                            self.last_task_report["changed_files"].append(path)
                        verification = verify_changed_file(self.workspace, path)
                        self.last_task_report["static_checks"].append(verification)
                        tool_result["verification"] = verification
                        mutation_needs_verification = True
                        if not verification.get("success"):
                            self.show_status(f"Automatic verification failed for {path}")
                if name == "run_command" and isinstance(tool_result, dict):
                    command = str(arguments.get("command", ""))
                    record = {
                        "command": command,
                        "return_code": tool_result.get("return_code"),
                        "success": tool_result.get("success") is True,
                        "meaningful_check": _is_meaningful_verification_command(command),
                    }
                    self.last_task_report["commands"].append(record)
                    if (
                        record["meaningful_check"]
                        and record["success"]
                        and record["return_code"] == 0
                    ):
                        mutation_needs_verification = False

                if local_json_mode and recovered:
                    readable_result = json.dumps(
                        tool_result, ensure_ascii=False, default=str
                    )[:MAX_TOOL_RESULT_CHARS]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"RESULT of tool {name} with arguments "
                            f"{json.dumps(arguments, ensure_ascii=False)}: "
                            f"{readable_result}\n\n"
                            "Continue the original task from this real result. "
                            "Choose the next distinct action as exactly one JSON object. "
                            "Do not repeat a successful action. If implementation is written, "
                            "run a relevant check and inspect the result before finishing."
                        ),
                    })
                elif duplicate_mutation:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
                else:
                    messages.append(tool_message)

            if duplicate_mutation:
                messages.append({
                    "role": "user",
                    "content": (
                        "A repeated file/folder mutation was blocked. Continue the original "
                        "task with the next distinct action; do not repeat the same JSON. "
                        "Write the actual implementation, then run a syntax check/test and "
                        "inspect the result. Finish only after verifying the workspace."
                    ),
                })

            # Recover from a local model repeating the same tool action. We bypass
            # the tool protocol, request a constrained file map, and accept only
            # explicitly requested relative paths.
            if (
                local_json_mode
                and not local_fallback_attempted
                and repeated_action_count >= 2
            ):
                missing = self._missing_expected_artifacts(task_context)
                if missing:
                    local_fallback_attempted = True
                    self.show_status("Repeated action detected; recovering missing deliverables")
                    recovered_files = self._generate_missing_files_locally(
                        client, task_context, _expected_artifact_paths(task_context)
                    )
                    if recovered_files:
                        summary = (
                            "Recovered from a repeated tool-call loop. Requested files were written "
                            "and the available verification command was run."
                        )
                        self.session_history.extend([
                            {"role": "user", "content": task},
                            {"role": "assistant", "content": summary},
                        ])
                        self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                        return self._format_final_report(summary)
                    return self._format_final_report(
                        "Local Codex detected a repeated tool-call loop and tried file-generation recovery, "
                        "but it could not produce all requested files with a passing verification command. "
                        "The task is NOT marked successful; inspect the reported checks and try a stronger model."
                    )

        limit_message = (
            f"Stopped after {self.max_steps} tool steps to avoid an infinite loop. "
            "Some work may remain; review the tool results above and ask me to continue."
        )
        self.session_history.extend([
            {"role": "user", "content": task},
            {"role": "assistant", "content": limit_message},
        ])
        self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
        return self._format_final_report(limit_message)

    def run(self, task):
        """Run one task with a single persistent tool-using coding agent."""
        print()
        print("=" * 60)
        print(" LOCAL CODEX · CODING SESSION")
        print("=" * 60)
        print(f" Workspace: {self.workspace}")
        provider = str(self.config.get("provider", "groq")).lower()
        if provider in {"nvidia", "nvidia_nim"}:
            model = self.config.get("nvidia_model", "qwen/qwen2.5-coder-32b-instruct")
        elif provider in {"ollama", "local"}:
            model = self.config.get("local_model", "qwen2.5-coder:1.5b")
        else:
            model = self.config.get("models", {}).get("coder", "not configured")
        print(f" Provider : {provider}")
        print(f" Model    : {model}")
        print(f" Tools    : {len(self.tools)}")
        print("-" * 60)

        try:
            result = self.integrate(task, [])
        finally:
            self.show_status("Done")
        return result
