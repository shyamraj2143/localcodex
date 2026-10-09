"""NVIDIA NIM client using the OpenAI-compatible chat completions API."""

from copy import deepcopy
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


class NvidiaClient:
    def __init__(
        self,
        model="poolside/laguna-xs-2.1",
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=0.1,
        max_tokens=8192,
        max_attempts=3,
        base_delay=1.5,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max(1, int(max_attempts))
        self.base_delay = max(0.1, float(base_delay))

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not found. Add it to the local-codex .env file. "
                f"Expected file: {os.path.join(BASE_DIR, '.env')}"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=180.0,
        )

    @staticmethod
    def _tools_unsupported(error):
        text = str(error).lower()
        return any(term in text for term in (
            "does not support tools",
            "tool calling is not supported",
            "does not support tool calling",
            "function calling is not supported",
            "tool use is not supported",
            "unsupported tool",
        ))

    @staticmethod
    def _json_tool_instruction(messages, tools):
        copied = deepcopy(messages)
        names = []
        for schema in tools or []:
            function = schema.get("function", {}) if isinstance(schema, dict) else {}
            name = function.get("name")
            if name:
                names.append(name)

        copied.append({
            "role": "user",
            "content": (
                "Continue the coding task. This model endpoint does not accept native "
                "tool schemas, so use a plain-text tool call. Return exactly one JSON "
                "object and no Markdown, in this format: "
                '{"name":"TOOL_NAME","arguments":{}}. '
                "Use only these registered tool names: "
                + ", ".join(names)
                + ". Arguments must match the tool's purpose. Do not invent tool names. "
                "If the task is not complete, choose the next real action, usually "
                "list_files first, then read/write/edit files and run checks. "
                "Do not claim an action was performed unless the tool result confirms it."
            ),
        })
        return copied

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

        for attempt in range(self.max_attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as error:
                if tools and self._tools_unsupported(error):
                    fallback_kwargs = dict(kwargs)
                    fallback_kwargs.pop("tools", None)
                    fallback_kwargs.pop("tool_choice", None)
                    fallback_kwargs["messages"] = self._json_tool_instruction(messages, tools)
                    try:
                        return self.client.chat.completions.create(**fallback_kwargs)
                    except Exception as fallback_error:
                        raise RuntimeError(
                            "NVIDIA model rejected native tool calling and the JSON "
                            f"fallback failed: {fallback_error}"
                        ) from fallback_error

                error_text = str(error)
                if "401" in error_text or "403" in error_text or "invalid_api_key" in error_text.lower():
                    raise RuntimeError(
                        "NVIDIA API authentication failed. Check NVIDIA_API_KEY in .env."
                    ) from error
                if attempt + 1 >= self.max_attempts:
                    raise RuntimeError(
                        f"NVIDIA API request failed for model '{self.model}': {error}"
                    ) from error

                time.sleep(self.base_delay * (2 ** attempt))

        raise RuntimeError("NVIDIA API request failed without a response.")
