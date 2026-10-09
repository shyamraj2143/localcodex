import os
import time
import random

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RateLimitError(Exception):
    pass


class LocalCodexLLM:

    def __init__(self, model):

        api_key = os.getenv("NVIDIA_API_KEY")

        if not api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY not found in .env"
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=180.0,
            max_retries=0
        )

        self.model = model

    def chat(self, messages, tools=None):

        kwargs = {
            "model": self.model,
            "messages": messages,

            # Lower reasoning overhead for our agent.
            "temperature": 0.1,
            "top_p": 0.95,

            "max_tokens": 8192,

            "stream": False
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        max_attempts = 4

        for attempt in range(max_attempts):

            try:

                return self.client.chat.completions.create(
                    **kwargs
                )

            except Exception as error:

                error_text = str(error)

                is_rate_limit = (
                    "503" in error_text
                    or "ResourceExhausted" in error_text
                    or "rate limit" in error_text.lower()
                    or "too many requests" in error_text.lower()
                )

                if not is_rate_limit:
                    raise

                if attempt == max_attempts - 1:

                    raise RateLimitError(
                        "NVIDIA API is currently overloaded "
                        "or rate limited. "
                        "Try again after some time."
                    ) from error

                # Exponential backoff:
                # 2s → 4s → 8s
                wait = (
                    2 ** attempt
                    + random.uniform(0, 1)
                )

                print(
                    f"\n⚠ NVIDIA rate limit. "
                    f"Retrying in {wait:.1f}s..."
                )

                time.sleep(wait)