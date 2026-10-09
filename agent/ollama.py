"""OpenAI-compatible client for a locally running Ollama server."""

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

            if (
                "connect" in message
                or "connection refused" in message
            ):
                raise RuntimeError(
                    "Ollama server connect nahi ho raha. "
                    "Ollama start karke dobara try karo."
                ) from error

            if "404" in message or "not found" in message:
                raise RuntimeError(
                    f"Model '{self.model}' nahi mila. "
                    f"PowerShell mein run karo: ollama pull {self.model}"
                ) from error

            raise RuntimeError(
                f"Ollama request failed: {error}"
            ) from error