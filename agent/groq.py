import os
import time

from openai import OpenAI
from dotenv import load_dotenv


# Always load .env from the local-codex project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE, override=True)


class GroqClient:

    def __init__(
        self,
        model,
        temperature=0.1,
        max_tokens=8192,
        max_attempts=3,
        base_delay=1.5
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.base_delay = base_delay

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                f"GROQ_API_KEY not found.\n"
                f"Expected .env file at:\n{ENV_FILE}"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def chat(self, messages, tools=None):

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error = None

        for attempt in range(self.max_attempts):

            try:
                return self.client.chat.completions.create(**kwargs)

            except Exception as error:

                last_error = error

                # Do not retry authentication errors
                error_text = str(error)

                if "401" in error_text or "invalid_api_key" in error_text:
                    raise

                if attempt == self.max_attempts - 1:
                    raise

                delay = self.base_delay * (2 ** attempt)

                print(
                    f"\n  [Groq retry "
                    f"{attempt + 1}/{self.max_attempts}] "
                    f"waiting {delay:.1f}s..."
                )

                time.sleep(delay)

        raise last_error