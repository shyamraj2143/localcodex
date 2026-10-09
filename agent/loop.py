import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent.groq import GroqClient
from agent.ollama import OllamaClient

import re
import uuid
from types import SimpleNamespace


def recover_text_tool_calls(content, available_tools):
    """Recover a tool call when a small model prints JSON as text."""
    if not content:
        return []

    text = content.strip()

    # Remove Markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Try the complete response, then the JSON object inside it.
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    data = None

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    if not isinstance(data, dict):
        return []

    # Support both {"name": ..., "arguments": ...}
    # and {"tool_calls": [...]} formats.
    calls = data.get("tool_calls")

    if not isinstance(calls, list):
        calls = [data]

    recovered = []

    for item in calls:
        if not isinstance(item, dict):
            continue

        function_data = item.get("function", {})
        if not isinstance(function_data, dict):
            function_data = {}

        name = item.get("name") or function_data.get("name")
        arguments = (
            item.get("arguments")
            if "arguments" in item
            else function_data.get("arguments", {})
        )

        # Never execute a tool that is not registered.
        if not isinstance(name, str) or name not in available_tools:
            continue

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue

        # Some small models emit [] instead of {}.
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
                    arguments=json.dumps(arguments),
                ),
            )
        )

    return recovered


class MultiModelAgent:

    def __init__(
        self,
        config,
        tools,
        tool_schemas,
        workspace,
        status=None
    ):

        self.config = config
        self.tools = tools
        self.tool_schemas = tool_schemas
        self.workspace = workspace
        self.status_callback = status

        self.models = config["models"]

        self.max_workers = config.get(
            "max_parallel_workers",
            4
        )

        self.max_steps = config.get(
            "max_agent_steps",
            12
        )

    # ========================================================
    # STATUS
    # ========================================================

    def show_status(self, message):

        if self.status_callback:
            self.status_callback(message)

    # ========================================================
    # CREATE GROQ CLIENT
    # ========================================================

    def get_client(self, role):
        """Create a client for the selected AI provider."""

        provider = str(
            self.config.get("provider", "groq")
        ).strip().lower()

        retry_config = self.config.get("retry", {})

        common = {
            "temperature": self.config.get(
                "temperature",
                0.1,
            ),
            "max_tokens": self.config.get(
                "max_tokens",
                8192,
            ),
        }

        # Local Ollama mode
        if provider in {"ollama", "local"}:
            return OllamaClient(
                model=self.config.get(
                    "local_model",
                    "qwen2.5-coder:1.5b",
                ),
                base_url=self.config.get(
                    "ollama_base_url",
                    "http://localhost:11434/v1",
                ),
                **common,
            )

        # Groq API mode
        if provider != "groq":
            raise ValueError(
                f"Unsupported provider: {provider}. "
                "Use 'groq' or 'ollama'."
            )

        if role not in self.models:
            raise KeyError(
                f"No Groq model configured for role: {role}"
            )

        return GroqClient(
            model=self.models[role],
            **common,
            max_attempts=retry_config.get(
                "max_attempts",
                3,
            ),
            base_delay=retry_config.get(
                "base_delay",
                1.5,
            ),
        )  
     

    # ========================================================
    # PROJECT SNAPSHOT
    # ========================================================

    def get_project_snapshot(self):

        try:

            result = self.tools["list_files"](
                self.workspace
            )

            return json.dumps(
                result,
                ensure_ascii=False
            )[:12000]

        except Exception as error:

            return f"Unable to scan project: {error}"

    # ========================================================
    # TASK CLASSIFICATION
    # ========================================================

    def classify_task(self, task):

        text = task.lower()

        complex_words = [
            "build",
            "create",
            "implement",
            "complete",
            "full",
            "application",
            "website",
            "backend",
            "frontend",
            "database",
            "authentication",
            "refactor",
            "deploy",
            "architecture",
            "integrate"
        ]

        medium_words = [
            "fix",
            "debug",
            "modify",
            "edit",
            "add",
            "feature",
            "bug",
            "error",
            "api",
            "component",
            "function"
        ]

        if any(
            word in text
            for word in complex_words
        ):

            return "COMPLEX"

        if any(
            word in text
            for word in medium_words
        ):

            return "MEDIUM"

        return "SIMPLE"

    # ========================================================
    # PLANNER
    # ========================================================

    def create_plan(self, task):

        self.show_status(
            "Planning"
        )

        client = self.get_client(
            "planner"
        )

        snapshot = self.get_project_snapshot()

        messages = [

            {
                "role": "system",
                "content": """
You are the lead software architect.

Analyze the user's coding task and
the current project.

Decide how many specialist workers
are actually needed.

Return ONLY valid JSON.

Format:

{
    "complexity": "SIMPLE|MEDIUM|COMPLEX",
    "workers": [
        {
            "role": "coder|fast|debugger|reviewer",
            "task": "specific task",
            "focus": "specific area"
        }
    ]
}

Rules:

SIMPLE:
Use 1 worker.

MEDIUM:
Use 2 workers.

COMPLEX:
Use 3 or 4 workers.

Do not create unnecessary workers.

Prefer independent work streams.

Workers should avoid modifying
the same files simultaneously.
"""
            },

            {
                "role": "user",
                "content": f"""
USER TASK:

{task}

CURRENT PROJECT:

{snapshot}
"""
            }

        ]

        try:

            response = client.chat(
                messages
            )

            content = (
                response.choices[0]
                .message
                .content
                or ""
            )

            start = content.find("{")
            end = content.rfind("}")

            if (
                start != -1
                and end != -1
            ):

                return json.loads(
                    content[
                        start:end + 1
                    ]
                )

        except Exception as error:

            self.show_status(
                "Planner fallback"
            )

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        complexity = self.classify_task(
            task
        )

        if complexity == "SIMPLE":

            workers = [
                {
                    "role": "coder",
                    "task": task,
                    "focus": "main implementation"
                }
            ]

        elif complexity == "MEDIUM":

            workers = [
                {
                    "role": "coder",
                    "task": task,
                    "focus": "implementation"
                },
                {
                    "role": "reviewer",
                    "task": task,
                    "focus": "review and find problems"
                }
            ]

        else:

            workers = [
                {
                    "role": "coder",
                    "task": task,
                    "focus": "implementation"
                },
                {
                    "role": "fast",
                    "task": task,
                    "focus": "project analysis"
                },
                {
                    "role": "debugger",
                    "task": task,
                    "focus": "bugs and edge cases"
                },
                {
                    "role": "reviewer",
                    "task": task,
                    "focus": "architecture and quality"
                }
            ]

        return {
            "complexity": complexity,
            "workers": workers
        }

    # ========================================================
    # EXECUTE TOOL
    # ========================================================

    def execute_tool(
        self,
        name,
        arguments
    ):

        if name not in self.tools:

            return {
                "success": False,
                "error": f"Unknown tool: {name}"
            }

        try:

            return self.tools[name](
                self.workspace,
                **arguments
            )

        except Exception as error:

            return {
                "success": False,
                "error": str(error)
            }

    # ========================================================
    # WORKER TOOL LOOP
    # ========================================================

    def worker_loop(
        self,
        worker,
        user_task
    ):

        role = worker["role"]

        self.show_status(
            f"{role.upper()} worker"
        )

        client = self.get_client(
            role
        )

        messages = [

            {
                "role": "system",
                "content": f"""
You are the {role} specialist
in a multi-agent software engineering
system.

Workspace:

{self.workspace}

Assigned task:

{worker["task"]}

Focus:

{worker["focus"]}

Inspect the project before making
decisions.

Use tools when necessary.

Do not make destructive changes.

Your output should contain:

1. What you inspected
2. Problems found
3. Root causes
4. Recommended solution
5. Files involved
6. Testing recommendations

You are an analysis specialist.
The final implementation will be
performed by the integration agent.
"""
            },

            {
                "role": "user",
                "content": user_task
            }

        ]

        for step in range(
            self.max_steps
        ):

            response = client.chat(
                messages,
                tools=self.tool_schemas
            )

            message = (
                response.choices[0]
                .message
            )

            # ------------------------------------------------
            # FINAL WORKER RESPONSE
            # ------------------------------------------------

            tool_calls = message.tool_calls
            recovered_text_call = False

            if not tool_calls:
                tool_calls = recover_text_tool_calls(
                    message.content,
                    self.tools,
                )
                recovered_text_call = bool(tool_calls)

            if not tool_calls:
                return message.content or "No report generated."

            # Keep exactly one assistant message in the conversation.
            # Recovered JSON calls need the same structure as native tool calls.
            if recovered_text_call:
                messages.append({
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
                })
            else:
                messages.append(message)

            # ------------------------------------------------
            # TOOL CALLS
            # ------------------------------------------------

            for tool_call in tool_calls:

                tool_name = (
                    tool_call
                    .function
                    .name
                )

                try:

                    arguments = json.loads(
                        tool_call
                        .function
                        .arguments
                    )

                except Exception:

                    arguments = {}

                result = self.execute_tool(
                    tool_name,
                    arguments
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id":
                            tool_call.id,
                        "content":
                            json.dumps(
                                result,
                                ensure_ascii=False
                            )
                    }
                )

        return (
            f"{role} worker reached "
            "maximum steps."
        )

    # ========================================================
    # PARALLEL WORKERS
    # ========================================================

    def run_parallel_workers(
        self,
        workers,
        task
    ):

        if not workers:

            return []

        worker_count = min(
            len(workers),
            self.max_workers
        )

        self.show_status(
            f"Starting {worker_count} "
            "workers in parallel"
        )

        results = []

        with ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:

            future_map = {}

            for worker in workers[
                :worker_count
            ]:

                future = executor.submit(
                    self.worker_loop,
                    worker,
                    task
                )

                future_map[
                    future
                ] = worker

            for future in as_completed(
                future_map
            ):

                worker = future_map[
                    future
                ]

                try:

                    result = future.result()

                except Exception as error:

                    result = (
                        f"Worker error: {error}"
                    )

                results.append(
                    {
                        "role":
                            worker["role"],
                        "result":
                            result
                    }
                )

                self.show_status(
                    f"{worker['role'].upper()} "
                    "completed"
                )

        return results

    # ========================================================
    # FINAL INTEGRATOR
    # ========================================================

    def integrate(
        self,
        task,
        reports
    ):

        self.show_status(
            "Final coding"
        )

        client = self.get_client(
            "coder"
        )

        reports_text = json.dumps(
            reports,
            ensure_ascii=False,
            indent=2
        )

        messages = [

            {
                "role": "system",
                "content": f"""
You are the primary autonomous
software engineering agent.

Current project:

{self.workspace}

You received reports from multiple
specialist agents.

Now actually complete the user's task.

IMPORTANT:

1. Inspect relevant files first.
2. Use the available tools.
3. Make actual code changes.
4. Do not merely explain the solution.
5. Preserve existing architecture.
6. Do not overwrite unrelated code.
7. Run appropriate tests.
8. If tests fail, diagnose and fix.
9. Continue until the task is complete.
10. Verify the final result.

You are the ONLY agent responsible
for final project modifications.
"""
            },

            {
                "role": "user",
                "content": f"""
USER TASK:

{task}

SPECIALIST REPORTS:

{reports_text}
"""
            }

        ]

        for step in range(
            self.max_steps
        ):

            response = client.chat(
                messages,
                tools=self.tool_schemas
            )

            message = (
                response.choices[0]
                .message
            )

            # ------------------------------------------------
            # FINAL RESPONSE
            # ------------------------------------------------

            tool_calls = message.tool_calls
            recovered_text_call = False

            if not tool_calls:
                tool_calls = recover_text_tool_calls(
                    message.content,
                    self.tools,
                )
                recovered_text_call = bool(tool_calls)

            if not tool_calls:
                return message.content or "Task completed."

            # Keep exactly one assistant message in the conversation.
            # Recovered JSON calls need the same structure as native tool calls.
            if recovered_text_call:
                messages.append({
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
                })
            else:
                messages.append(message)

            # ------------------------------------------------
            # EXECUTE TOOLS
            # ------------------------------------------------

            for tool_call in tool_calls:

                tool_name = (
                    tool_call
                    .function
                    .name
                )

                try:

                    arguments = json.loads(
                        tool_call
                        .function
                        .arguments
                    )

                except Exception:

                    arguments = {}

                # Status

                if tool_name == "list_files":

                    self.show_status(
                        "Scanning project"
                    )

                elif tool_name == "read_file":

                    self.show_status(
                        "Reading "
                        + arguments.get(
                            "path",
                            ""
                        )
                    )

                elif tool_name == "write_file":

                    self.show_status(
                        "Writing "
                        + arguments.get(
                            "path",
                            ""
                        )
                    )

                elif tool_name == "edit_file":

                    self.show_status(
                        "Editing "
                        + arguments.get(
                            "path",
                            ""
                        )
                    )

                elif tool_name == "search_files":

                    self.show_status(
                        "Searching "
                        + arguments.get(
                            "query",
                            ""
                        )
                    )

                elif tool_name == "run_command":

                    self.show_status(
                        "Running command"
                    )

                else:

                    self.show_status(
                        f"Using {tool_name}"
                    )

                result = self.execute_tool(
                    tool_name,
                    arguments
                )

                # Error

                if (
                    isinstance(result, dict)
                    and result.get("success")
                    is False
                ):

                    self.show_status(
                        "Error found - analyzing"
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id":
                            tool_call.id,
                        "content":
                            json.dumps(
                                result,
                                ensure_ascii=False
                            )
                    }
                )

        return (
            "Final coding agent reached "
            "maximum steps."
        )

    # ========================================================
    # MAIN RUN
    # ========================================================

    
    def run(self, task):
        """Plan and execute a task within configured worker limits."""

        plan = self.create_plan(task)

        complexity = plan.get(
            "complexity",
            self.classify_task(task),
        )

        workers = plan.get("workers", [])

        if not isinstance(workers, list):
            workers = []

        # Enforce the configured limit even if the model suggests more.
        workers = workers[:self.max_workers]

        print()
        print("=" * 50)
        print(" TASK EXECUTION")
        print("=" * 50)
        print(f" Complexity : {complexity}")
        print(f" Workers    : {len(workers)} / {self.max_workers}")
        print(f" Workspace  : {self.workspace}")
        print("-" * 50)

        reports = self.run_parallel_workers(workers, task)

        result = self.integrate(task, reports)

        self.show_status("Done")

        return result
