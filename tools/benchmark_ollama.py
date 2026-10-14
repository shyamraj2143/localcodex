"""Optional local-model latency comparison; parallel calls require explicit opt-in."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from agent.model_selection import discover_ollama_models
from agent.ollama import OllamaClient

ROOT = Path(__file__).resolve().parents[1]


def _request(client, prompt):
    started = time.perf_counter()
    try:
        response = client.chat([{"role": "user", "content": prompt}], tools=None)
        content = response.choices[0].message.content or ""
        return {"success": bool(content.strip()), "duration_seconds": time.perf_counter() - started,
                "response_chars": len(content)}
    except Exception as error:
        return {"success": False, "duration_seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {str(error)[:500]}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Installed model; defaults to config local_model")
    parser.add_argument("--include-parallel", action="store_true", help="Opt in to two simultaneous local generations; may exhaust RAM on CPU-only machines")
    args = parser.parse_args(argv)
    config = json.loads((ROOT / "config" / "config.json").read_text(encoding="utf-8"))
    model = args.model or config.get("local_model")
    installed = discover_ollama_models(config.get("ollama_base_url", "http://localhost:11434/v1"))
    if not installed:
        parser.error("Ollama is unavailable or has no installed models")
    if model not in installed:
        parser.error(f"Model is not installed: {model}")
    client = OllamaClient(model=model, base_url=config.get("ollama_base_url", "http://localhost:11434/v1"),
                          timeout=config.get("request_timeout_seconds", 90))
    prompt = "Reply with the single word ready. Do not call tools."
    sequential_start = time.perf_counter()
    sequential = [_request(client, prompt) for _ in range(2)]
    result = {"model": model, "sequential": {"requests": sequential,
              "elapsed_seconds": time.perf_counter() - sequential_start},
              "parallel": None, "note": "Latency benchmark only; no code changes or task completion are measured."}
    if args.include_parallel:
        parallel_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ollama-benchmark") as pool:
            parallel = list(pool.map(lambda _: _request(client, prompt), range(2)))
        result["parallel"] = {"requests": parallel, "elapsed_seconds": time.perf_counter() - parallel_start}
    print(json.dumps(result, indent=2))
    return 0 if all(row["success"] for row in sequential) and (result["parallel"] is None or all(row["success"] for row in result["parallel"]["requests"])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
