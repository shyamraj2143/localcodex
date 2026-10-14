
import os
import json
import sys
import re
import shutil

from dotenv import load_dotenv

from agent.dashboard import TerminalDashboard
from agent.loop import MultiModelAgent
from agent.model_selection import choose_provider, discover_ollama_models, validate_selected_model
from agent.orchestration import redact_secrets
from agent.request import classify_request
from tools.registry import get_tools, get_tool_schemas


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(
    os.path.join(BASE_DIR, ".env"),
    override=False,
)



# -------------------- CONSOLE UI --------------------

class UI:
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    @classmethod
    def color(cls, text, color):
        if os.getenv("NO_COLOR") or not sys.stdout.isatty():
            return str(text)
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def header(cls, workspace, provider, model, tool_count, execution_mode="OFFLINE", worker_count=0):
        project = os.path.basename(os.path.normpath(workspace)) or workspace
        print(cls.color("LOCAL CODEX", cls.BOLD + cls.CYAN)
              + f"  {provider.upper()} / {model}  [{execution_mode.upper()}]")
        print(f"Workspace: {project} ({workspace})\n")

    @classmethod
    def success(cls, message):
        print(cls.color(f"✓ {message}", cls.GREEN))

    @classmethod
    def error(cls, message):
        print(cls.color(f"✗ {message}", cls.RED))

    @classmethod
    def info(cls, message):
        print(cls.color(f"ℹ {message}", cls.CYAN))


# -------------------- CONFIG --------------------

def load_config():
    path = os.path.join(BASE_DIR, "config", "config.json")

    with open(path, "r", encoding="utf-8-sig") as file:
        config = json.load(file)

    mode = str(config.get("execution_mode", "auto")).strip().lower()
    if mode not in {"offline", "online", "auto"}:
        raise ValueError("execution_mode must be one of: offline, online, auto")
    config["execution_mode"] = mode

    # Optional local-model overrides live in each machine's untracked .env.
    # Accept either an Ollama server root or its OpenAI-compatible /v1 URL.
    ollama_host = os.getenv("OLLAMA_HOST", "").strip().rstrip("/")
    if ollama_host:
        config["ollama_base_url"] = ollama_host if ollama_host.endswith("/v1") else ollama_host + "/v1"
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        config["local_model"] = ollama_model
        options = config.get("startup_local_model_options", [])
        if isinstance(options, list) and ollama_model not in options:
            config["startup_local_model_options"] = [ollama_model, *options]

    # Protect against accidentally huge token limits.
    token_limit = config.get("max_tokens", 8192)
    if not isinstance(token_limit, int) or token_limit < 1:
        config["max_tokens"] = 8192
    else:
        config["max_tokens"] = min(token_limit, 16384)

    config["max_total_workers"] = max(2, min(int(config.get("max_total_workers", 3)), 8))
    config["max_parallel_tool_workers"] = max(1, min(
        int(config.get("max_parallel_tool_workers", config.get("max_parallel_workers", 2))),
        4, config["max_total_workers"] - 1,
    ))
    config["max_concurrent_llm_requests"] = max(1, min(
        int(config.get("max_concurrent_llm_requests", 1)), 2,
        config["max_total_workers"] - config["max_parallel_tool_workers"],
    ))
    config["max_agent_steps"] = max(1, min(int(config.get("max_agent_steps", 32)), 80))

    return config


def choose_startup_model(config):
    return choose_provider(config)


def offer_provider_recovery(agent, config, task):
    """Ask before retrying a failed model or explicitly choosing another provider."""
    report = agent.last_task_report or {}
    failed_requests = [item for item in report.get("llm_requests", []) if not item.get("success")]
    if report.get("verification_status") == "PASS" or not failed_requests:
        return None
    current = failed_requests[-1].get("provider", config.get("provider", "configured"))
    print(f"\n  {current} could not complete the task.")
    print("  [1] Retry the selected provider/model")
    print("  [2] Select another configured provider/model")
    print("  [3] Continue with an installed local Ollama model")
    try:
        choice = input("  Recovery choice [1/2/3, Enter to stop]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice == "1":
        return agent.run(task)
    if choice == "2":
        if config.get("execution_mode") == "offline":
            print("  Hosted providers are blocked in OFFLINE mode. Switch execution_mode to online first.")
            return None
        selected_provider = choose_provider(config)
        return agent.run(task) if selected_provider else None
    if choice == "3":
        local_models = discover_ollama_models(config.get("ollama_base_url", "http://localhost:11434/v1"))
        if not local_models:
            UI.error("Ollama has no available local models; no provider switch was made.")
            return None
        for index, name in enumerate(local_models, 1):
            print(f"  [{index}] {name}")
        try:
            selected = input("  Local model number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not selected.isdigit() or not 1 <= int(selected) <= len(local_models):
            UI.error("Invalid local model selection; provider was not changed.")
            return None
        config["provider"] = "ollama"
        config["local_model"] = local_models[int(selected) - 1]
        config["selected_model"] = config["local_model"]
        config["execution_mode"] = "offline"
        config["local_only"] = True
        config["hosted_api_enabled"] = False
        return agent.run(task)
    return None


# -------------------- SIMPLE CHAT --------------------

def simple_reply(text):
    """Answer simple conversational inputs without starting coding agents."""
    normalized = re.sub(r"[^\w\s']", "", text.lower()).strip()
    words = normalized.split()

    greetings = {
        "hi", "hello", "hey", "hii", "hiii",
        "namaste", "namaskar",
    }

    if normalized in greetings:
        return (
            "Hey! 👋 I'm Local Codex.\n\n"
            "I can help you inspect, create, debug and improve code.\n"
            "Tell me what you want to build."
        )

    if normalized in {"thanks", "thank you", "thx", "thankyou"}:
        return "You're welcome! 😊 What would you like to work on next?"

    if normalized in {"bye", "goodbye", "see you"}:
        return "See you! 👋 Type 'exit' whenever you want to close Codex."

    if normalized in {"help", "/help"}:
        return (
            "What I can do:\\n"
            "  • Inspect, create and edit files\\n"
            "  • Run commands and tests\\n"
            "  • Diagnose errors and iterate on fixes\\n"
            "  • Remember the recent coding conversation\\n\\n"
            "Commands: /help, /status, /clear, /exit\\n"
            "Example: Create a Python calculator and run its tests."
        )

    # Don't classify ordinary short conversational questions as coding tasks.
    if len(words) <= 2 and normalized in {
        "how are you", "who are you", "what is up",
    }:
        return "I'm ready to help with your coding tasks! 😊"

    return None


def answer_general_question(agent, question):
    """Answer general questions without exposing coding tools."""
    provider = str(
        agent.config.get("provider", "groq")
    ).lower()

    role = "fast" if provider == "groq" else "coder"
    client = agent.get_client(role)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful general-purpose assistant. "
                "Answer the user's question directly and clearly. "
                "Do not call tools. Do not output JSON or tool-call "
                "objects. If you cannot verify current information, "
                "say so honestly."
            ),
        },
        {
            "role": "user",
            "content": redact_secrets(question),
        },
    ]

    response = client.chat(messages)
    return (
        response.choices[0].message.content
        or "I couldn't generate an answer. Please try again."
    )


# -------------------- MAIN --------------------

def main():
    # Unicode UI borders/status messages otherwise crash under legacy Windows
    # code pages (for example cp1252 in redirected PowerShell sessions).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    try:
        config = load_config()
    except Exception as error:
        UI.error(f"Could not load config: {error}")
        return

    provider = choose_startup_model(config)
    if provider is None:
        return

    if provider == "ollama":
        model = config["local_model"]
    elif provider in {"nvidia", "nvidia_nim"}:
        model = config["nvidia_model"]
    else:
        model = config.get("models", {}).get("coder", "Not configured")

    workspace = os.path.realpath(os.environ.get("LOCAL_CODEX_WORKSPACE") or os.getcwd())
    config["selected_model"] = model

    try:
        tools = get_tools()
        schemas = get_tool_schemas()
    except Exception as error:
        UI.error(f"Could not load tools: {error}")
        return

    UI.header(workspace, provider, model, len(tools), config["execution_mode"], 0)

    dashboard = TerminalDashboard(
        mode=config.get("ui_mode", "auto"), workspace=workspace,
        provider=provider, model=model, execution_mode=config["execution_mode"],
        demo_mode=config.get("dashboard_demo_mode", True),
        language=config.get("dashboard_language", "english"),
        show_technical=config.get("dashboard_show_technical", False),
    )

    try:
        agent = MultiModelAgent(
            config=config,
            tools=tools,
            tool_schemas=schemas,
            workspace=workspace,
            status=dashboard.status,
            event_bus=dashboard.bus,
        )
    except Exception as error:
        UI.error(f"Agent startup failed: {error}")
        return

    agent.workspace_memory.refresh()
    dashboard.bus.emit("workspace.memory", **agent.workspace_memory.dashboard_state())
    if not agent.workspace_memory.enabled:
        UI.info(f"Workspace memory: {agent.workspace_memory.status}")

    UI.success("Codex is ready.")
    print("  Type /help for help or exit to quit.\n")

    while True:
        try:
            user_input = input(
                UI.color("  You  > ", UI.BOLD + UI.CYAN)
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "/exit","x"}:
            print("\nGoodbye! 👋")
            break

        command = user_input.lower()

        if command == "/help":
            print(simple_reply("/help"))
            print()
            continue

        if command == "/clear":
            agent.session_history.clear()
            agent.last_task_report.clear()
            agent.recent_tool_calls.clear()
            UI.success("Conversation context cleared. Project files were not changed.")
            print()
            continue

        if command == "/status":
            print()
            print(f"  Provider : {provider}")
            print(f"  Model    : {model}")
            print(f"  Workspace: {workspace}")
            print(f"  Tools    : {len(tools)}")
            print(f"  Memory   : {len(agent.session_history) // 2} recent turns")
            if agent.last_task_report:
                print(f"  Last task: {agent.last_task_report.get('task', 'unknown')}")
                print(f"  Verified : {agent.last_task_report.get('verification_status', 'unknown')}")
                missing = agent.last_task_report.get("missing_artifacts", [])
                if missing:
                    print(f"  Missing  : {', '.join(missing)}")
            print()
            continue

        if command == "/task":
            print("  Enter the complete task on multiple lines.")
            print("  Type END on a line by itself when you are finished.")
            task_lines = []
            try:
                while True:
                    line = input("  ... ")
                    if line.strip().upper() == "END":
                        break
                    task_lines.append(line)
            except (KeyboardInterrupt, EOFError):
                print("\n  Multi-line task cancelled.")
                print()
                continue

            user_input = "\n".join(task_lines).strip()
            if not user_input:
                UI.error("No task entered.")
                print()
                continue
            command = user_input.lower()

        # Handle greetings without running the coding agent.
        quick_response = simple_reply(user_input)
        if quick_response is not None:
            print()
            print(UI.color("  Codex", UI.BOLD + UI.GREEN))
            print("  " + quick_response.replace("\n", "\n  "))
            print()
            continue

        print()
        try:
            
            request_kind = classify_request(
                user_input,
                has_history=bool(agent.session_history or agent.last_task_report),
                has_project=bool(agent.workspace_context.files()),
            )
            if request_kind == "general":
                try:
                    answer = answer_general_question(
                        agent,
                        user_input,
                    )
                    print(
                        UI.color(
                            "\n  Codex",
                            UI.BOLD + UI.GREEN,
                        )
                    )
                    print("\n" + answer)
                except Exception as error:
                    UI.error(str(error))

                print()
                continue

            if request_kind == "explanation":
                try:
                    answer = agent.explain(user_input)
                    print(UI.color("\n  Codex", UI.BOLD + UI.GREEN))
                    print("\n" + answer)
                except Exception as error:
                    UI.error(str(error))
                print()
                continue

            valid_model, model_error = validate_selected_model(config, provider, model)
            if not valid_model:
                UI.error(model_error)
                print()
                continue

            result = agent.run(user_input)
            recovery_result = offer_provider_recovery(agent, config, user_input)
            if recovery_result is not None:
                result = recovery_result
            print(UI.color("\n  Codex", UI.BOLD + UI.GREEN))
            print()
            print(str(result).strip())
            report = agent.last_task_report or {}
            if report.get("duration_seconds") is not None:
                print(f"\n  Performance: task {report['duration_seconds']:.2f}s | "
                      f"LLM requests {len(report.get('llm_requests', []))} | "
                      f"tool operations {len(report.get('tool_durations', []))}")
        except Exception as error:
            UI.error(str(error))

        print()


if __name__ == "__main__":
    main()
