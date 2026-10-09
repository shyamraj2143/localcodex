"""Agentic coding loop for Local Codex.

Unlike the previous planner + parallel-worker pipeline, this loop keeps one
coding agent in control of inspect -> edit -> run -> diagnose -> verify. This
is more reliable for small local Ollama models and mirrors the core workflow
of an interactive coding agent.
"""

import json
import re
import uuid
from types import SimpleNamespace

from agent.groq import GroqClient
from agent.ollama import OllamaClient


MAX_TOOL_RESULT_CHARS = 16000
MAX_SESSION_TURNS = 8


def _extract_json_object(content):
    """Parse a JSON object from plain text or a fenced code block."""
    if not isinstance(content, str) or not content.strip():
        return None

    text = content.strip()
    text = re.sub(r"^\`\`\`(?:json)?\s*|\s*\`\`\`$", "", text, flags=re.IGNORECASE).strip()

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def recover_text_tool_calls(content, available_tools):
    """Convert JSON-shaped tool calls emitted as text into executable calls."""
    data = _extract_json_object(content)
    if not isinstance(data, dict):
        return []

    calls = data.get("tool_calls")
    if not isinstance(calls, list):
        calls = [data]

    recovered = []
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
                **common,
            )

        if provider != "groq":
            raise ValueError(
                f"Unsupported provider: {provider}. Use 'groq' or 'ollama'."
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

        tools_description = ", ".join(sorted(self.tools))
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
        for step in range(self.max_steps):
            self.show_status(f"Thinking · step {step + 1}/{self.max_steps}")
            response = client.chat(messages, tools=self.tool_schemas)
            message = response.choices[0].message
            native_calls = getattr(message, "tool_calls", None) or []
            tool_calls = list(native_calls)
            recovered = False

            if not tool_calls:
                tool_calls = recover_text_tool_calls(message.content, self.tools)
                recovered = bool(tool_calls)

            if not tool_calls:
                if _is_tool_call_shaped_json(getattr(message, "content", "")) and malformed_tool_retries < 2:
                    malformed_tool_retries += 1
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "That looked like a tool call, but it was not executable. "
                            f"Use only these exact registered tools: {tools_description}. "
                            "Return a valid native tool call, or JSON with keys name and arguments. "
                            "Arguments must be an object. Continue the task; do not stop here."
                        ),
                    })
                    continue

                final_text = (getattr(message, "content", None) or "").strip()
                if not final_text:
                    final_text = (
                        "I couldn't complete the task because the model returned no final response. "
                        "Try again or use a stronger coding model."
                    )
                self.session_history.extend([
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": final_text},
                ])
                self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
                return final_text

            if recovered:
                messages.append(self._assistant_tool_call_message(tool_calls))
            else:
                messages.append(message)

            # Execute every requested tool and feed real results back to the model.
            for tool_call in tool_calls:
                messages.append(self._tool_message(tool_call))

        limit_message = (
            f"Stopped after {self.max_steps} tool steps to avoid an infinite loop. "
            "Some work may remain; review the tool results above and ask me to continue."
        )
        self.session_history.extend([
            {"role": "user", "content": task},
            {"role": "assistant", "content": limit_message},
        ])
        self.session_history = self.session_history[-(MAX_SESSION_TURNS * 2):]
        return limit_message

    def run(self, task):
        """Run one task with a single persistent tool-using coding agent."""
        print()
        print("=" * 60)
        print(" LOCAL CODEX · CODING SESSION")
        print("=" * 60)
        print(f" Workspace: {self.workspace}")
        print(f" Provider : {self.config.get('provider', 'groq')}")
        print(f" Tools    : {len(self.tools)}")
        print("-" * 60)

        try:
            result = self.integrate(task, [])
        finally:
            self.show_status("Done")
        return result
