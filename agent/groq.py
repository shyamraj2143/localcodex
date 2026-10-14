import os
from openai import OpenAI
from dotenv import load_dotenv
import time

from agent.provider_support import credential_present, is_auth_error, is_transient, provider_error, retry_delay


# Always load .env from the local-codex project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ENV_FILE = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_FILE, override=False)


class GroqClient:

    def __init__(
        self,
        model,
        temperature=0.1,
        max_tokens=8192,
        max_attempts=3,
        base_delay=1.5,
        timeout=90,
        on_retry=None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max(1, min(int(max_attempts), 5))
        self.base_delay = max(0.1, min(float(base_delay), 30))
        self.timeout = max(1, min(float(timeout), 300))
        self.on_retry = on_retry

        api_key = os.getenv("GROQ_API_KEY")

        if not credential_present("GROQ_API_KEY"):
            raise RuntimeError(f"GROQ_API_KEY not found. Configure it in the project .env file: {ENV_FILE}")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=self.timeout,
            max_retries=0,
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

        for attempt in range(self.max_attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as error:
                if is_auth_error(error) or not is_transient(error) or attempt + 1 >= self.max_attempts:
                    raise provider_error("Groq", self.model, error) from error
                delay = retry_delay(error, attempt, self.base_delay)
                callback = getattr(self, "on_retry", None)
                if callback:
                    callback(attempt + 1, self.max_attempts - 1, delay, error)
                else:
                    print(f"Groq temporary request error; retry {attempt + 1}/{self.max_attempts - 1} in {delay:.1f}s.")
                time.sleep(delay)

        raise RuntimeError("Groq request stopped without a response.")
