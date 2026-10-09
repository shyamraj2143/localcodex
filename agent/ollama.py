"""OpenAI-compatible client for a locally running Ollama server."""

from copy import deepcopy

from openai import OpenAI


class OllamaClient:
    def __init__(
        self,
        model="qwen2.5-coder:1.5b",
        base_url="http://localhost:11434/v1",
        temperature=0.1,
        max_tokens=8192,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=300.0,
        )

    def chat(self, messages, tools=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            return self.client.chat.completions.create(**kwargs)

        except Exception as error:
            message = str(error).lower()

            # Some Ollama models (including deepseek-coder:1.3b) do not
            # support native function/tool calling. Retry without API tool
            # schemas and ask the model to emit a JSON tool call as plain text.
            if tools and (
                "does not support tools" in message
                or "tool calling is not supported" in message
                or "does not support tool calling" in message
            ):
                fallback_messages = deepcopy(messages)
                tool_names = []
                for schema in tools:
                    function = schema.get("function", {}) if isinstance(schema, dict) else {}
                    name = function.get("name")
                    if name:
                        tool_names.append(name)

                fallback_messages.append({
                    "role": "user",
                    "content": (
                        "This model cannot use native tool calling. Continue the task "
                        "by returning exactly one plain JSON object for the next tool "
                        "action, with no Markdown or explanation. Format: "
                        '{"name":"TOOL_NAME","arguments":{}}. '
                        "Use only one of these registered tool names: "
                        + ", ".join(tool_names)
                        + ". Do not invent tool names or arguments. If the task is "
                        "already complete, return a concise final answer instead."
                    ),
                })
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("tools", None)
                fallback_kwargs.pop("tool_choice", None)
                fallback_kwargs["messages"] = fallback_messages

                try:
                    return self.client.chat.completions.create(**fallback_kwargs)
                except Exception as fallback_error:
                    raise RuntimeError(
                        "Ollama model does not support native tools, and the "
                        f"plain-JSON fallback also failed: {fallback_error}"
                    ) from fallback_error

            if "connect" in message or "connection refused" in message:
                raise RuntimeError(
                    "Ollama server connect nahi ho raha. "
                    "Ollama start karke dobara try karo."
                ) from error

            if "404" in message or "not found" in message:
                raise RuntimeError(
                    f"Model '{self.model}' nahi mila. "
                    f"PowerShell mein run karo: ollama pull {self.model}"
                ) from error

            raise RuntimeError(f"Ollama request failed: {error}") from error
