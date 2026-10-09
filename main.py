
import os
import json
import sys
import itertools
import re

from dotenv import load_dotenv

from agent.loop import MultiModelAgent
from tools.registry import get_tools, get_tool_schemas


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(
    os.path.join(BASE_DIR, ".env"),
    override=True,
)

spinner = itertools.cycle(["|", "/", "-", "\\"])


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
    def header(cls, workspace, provider, model, tool_count):
        width = 62

        print()
        print(cls.color("╭" + "─" * width + "╮", cls.CYAN))
        print(
            cls.color("│", cls.CYAN)
            + cls.color(
                "  LOCAL CODEX",
                cls.BOLD + cls.CYAN,
            )
            + " " * (width - 15)
            + cls.color("│", cls.CYAN)
        )
        print(cls.color("├" + "─" * width + "┤", cls.CYAN))
        print(f"  Provider : {provider.upper()}")
        print(f"  Model    : {model}")
        print(f"  Tools    : {tool_count} available")
        print(f"  Workspace: {workspace}")
        print(cls.color("╰" + "─" * width + "╯", cls.CYAN))
        print()

    @classmethod
    def success(cls, message):
        print(cls.color(f"✓ {message}", cls.GREEN))

    @classmethod
    def error(cls, message):
        print(cls.color(f"✗ {message}", cls.RED))

    @classmethod
    def info(cls, message):
        print(cls.color(f"ℹ {message}", cls.CYAN))


def show_status(message):
    icon = next(spinner)
    sys.stdout.write(f"\r\033[2K{icon} {message}...")
    sys.stdout.flush()


def clear_status():
    sys.stdout.write("\r\033[2K")
    sys.stdout.flush()


# -------------------- CONFIG --------------------

def load_config():
    path = os.path.join(BASE_DIR, "config", "config.json")

    with open(path, "r", encoding="utf-8-sig") as file:
        config = json.load(file)

    # Protect against accidentally huge token limits.
    token_limit = config.get("max_tokens", 8192)
    if not isinstance(token_limit, int) or token_limit < 1:
        config["max_tokens"] = 8192
    else:
        config["max_tokens"] = min(token_limit, 16384)

    config["max_parallel_workers"] = max(
        1,
        min(int(config.get("max_parallel_workers", 4)), 4),
    )

    return config


def choose_provider(config):
    configured = str(config.get("provider", "groq")).lower()
    default = "2" if configured in {"ollama", "local"} else "1"

    print("  Select provider")
    print("  [1] Groq API  — online")
    print(
        "  [2] Ollama    — local "
        f"({config.get('local_model', 'qwen2.5-coder:1.5b')})"
    )

    try:
        choice = input(f"\n  Choose [1/2, Enter={default}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return None

    choice = choice or default

    if choice == "1":
        config["provider"] = "groq"
    elif choice == "2":
        config["provider"] = "ollama"
    else:
        UI.error("Invalid choice.")
        return None

    return config["provider"]


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
            "What I can do:\n"
            "  • Create or edit project files\n"
            "  • Debug errors\n"
            "  • Explain existing code\n"
            "  • Run project commands\n\n"
            "Example: Create a Python calculator in calculator.py"
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
            "content": question,
        },
    ]

    response = client.chat(messages)
    return (
        response.choices[0].message.content
        or "I couldn't generate an answer. Please try again."
    )


def looks_like_general_question(text):
    text = text.lower().strip()

    coding_terms = (
        "code", "python", "javascript", "program", "function",
        "class ", "debug", "error", "bug", "file", "folder",
        "project", "website", "app ", "application", "api",
        "database", "terminal", "script", "implement", "refactor",
        "repository", "html", "css", "react", "node.js",
        "install", "compile", "exception", "stack trace",
    )

    if any(term in text for term in coding_terms):
        return False

    question_starters = (
        "who ", "what ", "when ", "where ", "why ",
        "is ", "are ", "was ", "were ", "which ",
        "tell me about ", "explain ",
    )

    return text.startswith(question_starters)

# -------------------- MAIN --------------------

def main():
    try:
        config = load_config()
    except Exception as error:
        UI.error(f"Could not load config: {error}")
        return

    provider = choose_provider(config)
    if provider is None:
        return

    if provider == "ollama":
        model = config.get("local_model", "qwen2.5-coder:1.5b")
    else:
        model = config.get("models", {}).get("coder", "Not configured")

    workspace = os.getcwd()

    try:
        tools = get_tools()
        schemas = get_tool_schemas()
    except Exception as error:
        UI.error(f"Could not load tools: {error}")
        return

    UI.header(workspace, provider, model, len(tools))

    try:
        agent = MultiModelAgent(
            config=config,
            tools=tools,
            tool_schemas=schemas,
            workspace=workspace,
            status=show_status,
        )
    except Exception as error:
        UI.error(f"Agent startup failed: {error}")
        return

    UI.success("Codex is ready.")
    print("  Type /help for help or exit to quit.\n")

    while True:
        try:
            user_input = input(
                UI.color("  You  > ", UI.BOLD + UI.CYAN)
            ).strip()
        except (KeyboardInterrupt, EOFError):
            clear_status()
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "/exit"}:
            print("\nGoodbye! 👋")
            break

        if user_input.lower() == "/help":
            print(simple_reply("/help"))
            print()
            continue

        # Handle greetings without running the planner/workers.
        quick_response = simple_reply(user_input)
        if quick_response is not None:
            print()
            print(UI.color("  Codex", UI.BOLD + UI.GREEN))
            print("  " + quick_response.replace("\n", "\n  "))
            print()
            continue

        print()
        try:
            
            if looks_like_general_question(user_input):
                try:
                    answer = answer_general_question(
                        agent,
                        user_input,
                    )
                    clear_status()
                    print(
                        UI.color(
                            "\n  Codex",
                            UI.BOLD + UI.GREEN,
                        )
                    )
                    print("\n" + answer)
                except Exception as error:
                    clear_status()
                    UI.error(str(error))

                print()
                continue

            result = agent.run(user_input)
            clear_status()

            print(UI.color("\n  Codex", UI.BOLD + UI.GREEN))
            print()
            print(str(result).strip())
        except Exception as error:
            clear_status()
            UI.error(str(error))

        print()


if __name__ == "__main__":
    main()
