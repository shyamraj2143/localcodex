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
        force_json_tools=True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        # Small local models often emit malformed native tool calls. A strict,
        # one-action JSON protocol is easier to recover and test.
        self.force_json_tools = bool(force_json_tools)

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=300.0,
        )

    @staticmethod
    def _tool_instructions(messages, tools):
        copied = deepcopy(messages)
        descriptions = []
        for schema in tools or []:
            function = schema.get("function", {}) if isinstance(schema, dict) else {}
            name = function.get("name")
            if not name:
                continue
            parameters = function.get("parameters", {})
            descriptions.append({
                "name": name,
                "description": function.get("description", ""),
                "parameters": parameters,
            })

        copied.append({
            "role": "user",
            "content": (
                "LOCAL CODING AGENT TOOL PROTOCOL. Return exactly ONE valid JSON "
                "object, with no Markdown, code fences, or explanation. Use this shape: "
                '{"name":"registered_tool_name","arguments":{}}. '
                "Choose exactly one next action per response. Tool catalogue (names, "
                "purpose, and required arguments): "
                + __import__("json").dumps(descriptions, ensure_ascii=False)
                + ". Rules: use only listed tools; arguments must match their schema; "
                "do not invent tool names or argument keys. First inspect files for an "
                "existing project. Then implement the smallest complete change. After "
                "writing code, run an appropriate syntax check/test and inspect the "
                "result. Read tool results carefully. Never repeat a successful action "
                "unless the result shows it failed. Never claim work is done without "
                "tool evidence. If no action is needed and the task is verified, answer "
                "normally with a concise summary instead of JSON."
            ),
        })
        return copied

    @staticmethod
    def _native_tools_unsupported(message):
        message = message.lower()
        return any(term in message for term in (
            "does not support tools",
            "tool calling is not supported",
            "does not support tool calling",
            "function calling is not supported",
            "tool use is not supported",
            "unsupported tool",
        ))

    def chat(self, messages, tools=None):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Prefer a single plain-JSON action for local models by default. This
        # avoids malformed native tool-call payloads common with small models.
        if tools and self.force_json_tools:
            kwargs["messages"] = self._tool_instructions(messages, tools)
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as error:
                raise RuntimeError(f"Ollama request failed: {error}") from error

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            return self.client.chat.completions.create(**kwargs)

        except Exception as error:
            message = str(error).lower()

            # Compatibility fallback for models that reject native function calling.
            if tools and self._native_tools_unsupported(message):
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("tools", None)
                fallback_kwargs.pop("tool_choice", None)
                fallback_kwargs["messages"] = self._tool_instructions(messages, tools)
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
