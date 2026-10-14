"""Startup provider/model discovery and interactive selection."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

from agent.provider_support import credential_present


def discover_ollama_models(base_url=None, timeout=1.5):
    """Return installed Ollama model tags, trying the HTTP API and CLI.

    A successful but empty API response is not treated as authoritative: the
    CLI is also checked because some proxy/compatibility endpoints return an
    empty payload even while the local Ollama CLI has models installed.
    """
    base_url = base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434/v1"
    root = str(base_url).strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    discovered = set()
    try:
        request = urllib.request.Request(root + "/api/tags", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=max(0.1, float(timeout))) as response:
            payload = json.loads(response.read(1_000_000).decode("utf-8"))
        models = payload.get("models", []) if isinstance(payload, dict) else []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name.strip():
                discovered.add(name.strip())
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        pass

    if not discovered:
        try:
            completed = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                                       timeout=max(0.1, float(timeout)), check=False)
            if completed.returncode == 0:
                rows = completed.stdout.splitlines()[1:]
                for line in rows:
                    columns = line.split()
                    if columns and columns[0].lower() not in {"name", "model"}:
                        discovered.add(columns[0].strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return sorted(discovered)


def configured_models(config, provider):
    if provider in {"ollama", "local"}:
        return list(config.get("ollama_models", []))
    models = config.get("models", {})
    if provider in {"nvidia", "nvidia_nim"}:
        values = [config.get(f"nvidia_{role}_model") for role in ("coder", "planner", "debugger", "reviewer", "fast")]
        values.append(config.get("nvidia_model"))
        return list(dict.fromkeys(value for value in values if isinstance(value, str) and value.strip()))
    keys = ("coder", "planner", "debugger", "reviewer", "fast")
    values = [models.get(key) for key in keys if isinstance(models, dict)]
    return list(dict.fromkeys(x for x in values if isinstance(x, str) and x.strip()))


def validate_selected_model(config, provider, model, discovered=None):
    if provider in {"ollama", "local"}:
        installed = discovered if discovered is not None else discover_ollama_models(
            config.get("ollama_base_url", "http://localhost:11434/v1"))
        if not installed:
            return False, "Ollama is unavailable or has no installed models. Start Ollama and retry."
        if model not in installed:
            return False, f"Selected Ollama model '{model}' is no longer installed. Restart and select an installed model."
    elif model not in configured_models(config, provider):
        return False, f"Selected {provider} model is not configured."
    return True, None


def choose_provider(config, input_fn=input, output=print, models=None):
    """Select an allowed provider and a real configured/installed model.

    Offline mode offers two configured local models and visibly marks configured
    hosted choices as requiring explicit confirmation to switch online.
    """
    offline = config.get("execution_mode") == "offline"
    local = models if models is not None else discover_ollama_models(config.get("ollama_base_url", "http://localhost:11434/v1"))
    default_provider = str(config.get("provider", "ollama" if offline else "groq")).lower()
    if config.get("execution_mode") == "hybrid":
        # Keep this startup menu intentionally fixed at exactly four choices.
        # Hosted options remain visible when keys are missing, but cannot be selected
        # until the corresponding key is configured.
        choices = [
            ("nvidia", "qwen/qwen2.5-coder-32b-instruct", "NVIDIA coding model", "NVIDIA_API_KEY"),
            ("groq", "qwen/qwen3.8-27b", "Groq coding model", "GROQ_API_KEY"),
            ("ollama", "deepseek-coder:1.3b", "DeepSeek local model", None),
            ("ollama", "qwen2.5-coder:3b", "Qwen2.5-Coder 3B local model", None),
        ]
        output("Choose one of exactly four coding models:")
        for index, (_provider, model_name, label, key_name) in enumerate(choices, 1):
            suffix = ""
            if key_name and not _has_credential(key_name):
                suffix = f" · {key_name} missing"
            if not key_name and model_name not in local:
                suffix = " · not installed locally"
            output(f"  [{index}] {label} — {model_name}{suffix}")
        try:
            raw_choice = input_fn("Coding model [4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        choice_index = 4 if not raw_choice else int(raw_choice) if raw_choice.isdigit() else 0
        if not 1 <= choice_index <= 4:
            output("Invalid choice. Enter exactly 1, 2, 3, or 4.")
            return None
        selected_provider, selected_model, _label, key_name = choices[choice_index - 1]
        if key_name and not _has_credential(key_name):
            output(f"Cannot select this option: configure {key_name} in .env first.")
            return None
        if selected_provider == "ollama" and selected_model not in local:
            output(f"Local model '{selected_model}' is not installed. Run: ollama pull {selected_model}")
            return None
        config["provider"] = selected_provider
        config["selected_model"] = selected_model
        # A local model selection must never depend on a hosted API key.
        # Hosted providers are available only when the user explicitly selects
        # one of the hosted menu entries; local runs are deterministic/local-only.
        if selected_provider == "ollama":
            config["execution_mode"] = "offline"
            config["local_model"] = selected_model
            config["local_only"] = True
            config["hosted_api_enabled"] = False
            config["task_architect_enabled"] = False
            config["hosted_coder_escalation"] = False
            config["provider_fallback_enabled"] = False
        elif selected_provider == "nvidia":
            config["local_only"] = False
            config["hosted_api_enabled"] = True
            config["execution_mode"] = "online"
            config["nvidia_model"] = selected_model
            config["hosted_coder_escalation"] = False
        else:
            config["execution_mode"] = "online"
            config.setdefault("models", {})["coder"] = selected_model
            config["hosted_coder_escalation"] = False
        return selected_provider
    if config.get("execution_mode") == "hybrid":
        if not local:
            output("HYBRID mode requires an installed local Ollama model. Start Ollama and install the configured model.")
            return None
        configured_default = str(config.get("local_model") or "").strip()
        selected_model = configured_default if configured_default in local else local[0]
        config["provider"] = "ollama"
        config["local_model"] = selected_model
        config["selected_model"] = selected_model
        config["local_only"] = False
        config["hosted_api_enabled"] = True
        output("Execution mode HYBRID: Ollama will write and run code; Groq may help interpret complex requests.")
        output(f"Local coding model: {selected_model}")
        if not _has_credential("GROQ_API_KEY"):
            output("GROQ_API_KEY is not configured; request understanding will use the local-only fallback.")
        return "ollama"
    if offline:
        output("Execution mode OFFLINE. NVIDIA/Groq selections require an explicit switch to ONLINE.")
        configured_local = config.get("startup_local_model_options") or []
        # Always offer the configured default when it is installed, then any
        # configured alternatives, then other discovered local models. This
        # avoids a broken [0] default when startup options are stale/incomplete.
        default_local = str(config.get("local_model") or "").strip()
        ordered_local = ([default_local] if default_local in local else [])
        ordered_local.extend(name for name in configured_local if name in local and name not in ordered_local)
        ordered_local.extend(name for name in local if name not in ordered_local)
        local_options = ordered_local[:4]
        choices = [("ollama", name) for name in local_options]
        if configured_models(config, "nvidia"):
            choices.append(("nvidia", configured_models(config, "nvidia")[0]))
        if configured_models(config, "groq"):
            choices.append(("groq", configured_models(config, "groq")[0]))
        if not choices:
            output("No configured local model is installed. Start Ollama and install one of the configured local models.")
            return None
        default_local = config.get("local_model")
        default_index = next((i + 1 for i, choice in enumerate(choices)
                              if choice == ("ollama", default_local)), 1)
        output("Choose provider/model:")
        for index, (choice_provider, choice_model) in enumerate(choices, 1):
            suffix = " (OFFLINE)" if choice_provider == "ollama" else " (switch to ONLINE)"
            if choice_provider == "nvidia" and not _has_credential("NVIDIA_API_KEY"):
                suffix += " · NVIDIA_API_KEY missing"
            if choice_provider == "groq" and not _has_credential("GROQ_API_KEY"):
                suffix += " · GROQ_API_KEY missing"
            output(f"  [{index}] {choice_provider.upper()} · {choice_model}{suffix}")
        output(f"  [0] Reuse default: {default_local} (Ollama)")
        try:
            choice = input_fn(f"Model [{default_index}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice or choice == "0":
            selected_provider, selected_model = "ollama", default_local
            if selected_model not in local:
                output(f"Configured default model '{selected_model}' is not installed.")
                return None
        elif choice.isdigit() and 1 <= int(choice) <= len(choices):
            selected_provider, selected_model = choices[int(choice) - 1]
        else:
            output("Invalid provider/model selection.")
            return None
        if selected_provider != "ollama":
            if selected_provider == "nvidia" and not _has_credential("NVIDIA_API_KEY"):
                output("NVIDIA_API_KEY is missing; configure credentials before selecting this model.")
                return None
            if selected_provider == "groq" and not _has_credential("GROQ_API_KEY"):
                output("GROQ_API_KEY is missing; configure credentials before selecting this model.")
                return None
            try:
                confirmation = input_fn(f"This uses {selected_provider.upper()} over the network and switches this session to ONLINE. Continue? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if confirmation not in {"y", "yes"}:
                output("Selection cancelled; execution mode remains OFFLINE.")
                return None
            config["execution_mode"] = "online"
            config["local_only"] = False
            config["hosted_api_enabled"] = True
        config["provider"] = selected_provider
        config["selected_model"] = selected_model
        if selected_provider == "ollama":
            config["local_model"] = selected_model
        elif selected_provider == "nvidia":
            config["nvidia_model"] = selected_model
        else:
            config.setdefault("models", {})["coder"] = selected_model
        return selected_provider
    providers = ["ollama"] if offline else [p for p in ("ollama", "groq", "nvidia") if (
        p == "ollama" and local or p == "groq" and _groq_configured(config) or p == "nvidia" and _nvidia_configured(config))]
    if not providers:
        output("No provider is available: install an Ollama model or configure a hosted provider and credentials.")
        return None
    if offline:
        provider = "ollama"
    else:
        output("Select provider:")
        for index, name in enumerate(providers, 1):
            configured = local if name == "ollama" else configured_models(config, name)
            output(f"  [{index}] {name.upper()} ({', '.join(configured)})")
        default_index = providers.index(default_provider) + 1 if default_provider in providers else 1
        try:
            choice = input_fn(f"Provider [{default_index}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice:
            provider = providers[default_index - 1]
        elif choice.isdigit() and 1 <= int(choice) <= len(providers):
            provider = providers[int(choice) - 1]
        else:
            output("Invalid provider selection.")
            return None
    available = local if provider == "ollama" else configured_models(config, provider)
    if not available:
        output("Ollama is unavailable or has no installed models. Start Ollama and install a model, then retry.")
        return None
    configured_default = (config.get("local_model") if provider == "ollama" else
                           config.get("nvidia_model") if provider == "nvidia" else
                           config.get("models", {}).get("coder"))
    if configured_default not in available:
        configured_default = available[0]
    output(f"Models for {provider.upper()}:")
    for index, model in enumerate(available, 1):
        output(f"  [{index}] {model}")
    output(f"  [0] Reuse default: {configured_default}")
    try:
        choice = input_fn(f"Model [0]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice or choice == "0":
        selected = configured_default
    elif choice.isdigit() and 1 <= int(choice) <= len(available):
        selected = available[int(choice) - 1]
    else:
        output("Invalid model selection.")
        return None
    config["provider"] = provider
    if provider == "ollama":
        config["local_model"] = selected
        config["selected_model"] = selected
    elif provider == "nvidia":
        config["nvidia_model"] = selected
        config["selected_model"] = selected
    else:
        config.setdefault("models", {})["coder"] = selected
        config["selected_model"] = selected
    return provider


def _groq_configured(config):
    return bool(config.get("models", {}).get("coder") and _has_credential("GROQ_API_KEY"))


def _nvidia_configured(config):
    return bool((config.get("nvidia_model") or config.get("nvidia_coder_model")) and _has_credential("NVIDIA_API_KEY"))


def _has_credential(name):
    return credential_present(name)
