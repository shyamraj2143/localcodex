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
        timeout=300,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        # Small local models often emit malformed native tool calls. A strict,
        # one-action JSON protocol is easier to recover and test.
        self.force_json_tools = bool(force_json_tools)
        self.timeout = max(1, min(float(timeout), 600))

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=self.timeout,
            max_retries=0,
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

    def _request_error(self, error):
        message = str(error)
        lowered = message.lower()
        if any(phrase in lowered for phrase in ("context length", "context window", "maximum context", "too many tokens")):
            return RuntimeError(f"Ollama model '{self.model}' exceeded its context limit. Reduce workspace context or use a larger-context model.")
        if "404" in lowered or "not found" in lowered or "model does not exist" in lowered:
            return RuntimeError(
                f"Local Ollama model '{self.model}' is not installed. "
                "Install it separately when internet access is allowed, then retry."
            )
        if "connect" in lowered or "connection refused" in lowered:
            return RuntimeError(
                "Ollama is unavailable at the configured local endpoint. Start Ollama and verify the local service, then retry."
            )
        if "timeout" in lowered or "timed out" in lowered or "deadline exceeded" in lowered:
            return RuntimeError(
                f"Ollama model '{self.model}' timed out after {self.timeout:g}s. "
                "The request timeout is configurable with request_timeout_seconds in config/config.json; "
                "on a CPU-only machine, keep workspace context compact and avoid running parallel generations."
            )
        return RuntimeError(f"Ollama request failed: {message[:1200]}")

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
                raise self._request_error(error) from error

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

            raise self._request_error(error) from error
