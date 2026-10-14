import json
from unittest.mock import patch

from agent.model_selection import choose_provider, discover_ollama_models, validate_selected_model


def test_offline_selection_uses_only_installed_ollama_models_and_updates_config():
    config = {"execution_mode": "offline", "provider": "ollama", "local_model": "coder:small",
              "secondary_local_model": "coder:large"}
    assert choose_provider(config, input_fn=lambda prompt: "2", output=lambda value: None,
                           models=["coder:small", "coder:large"]) == "ollama"
    assert config["local_model"] == "coder:large"
    assert config["selected_model"] == "coder:large"


def test_empty_ollama_list_gracefully_refuses_start():
    output = []
    config = {"execution_mode": "offline", "provider": "ollama", "local_model": "not-installed"}
    assert choose_provider(config, output=output.append, models=[]) is None
    assert "No configured local model is installed" in output[-1]
    assert config["provider"] == "ollama"


def test_online_selection_lists_only_providers_with_configuration_and_credentials():
    config = {"execution_mode": "online", "provider": "groq", "models": {"coder": "groq-model"}}
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key", "NVIDIA_API_KEY": ""}):
        assert choose_provider(config, input_fn=lambda prompt: "1", output=lambda value: None,
                               models=[]) == "groq"
    assert config["selected_model"] == "groq-model"


def test_ollama_discovery_parses_tags_api_and_handles_unavailable():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit): return json.dumps({"models": [{"name": "coder:latest"}]}).encode()
    with patch("urllib.request.urlopen", return_value=Response()):
        assert discover_ollama_models(timeout=0.1) == ["coder:latest"]
    with patch("urllib.request.urlopen", side_effect=OSError("offline")), \
         patch("subprocess.run", side_effect=OSError("missing cli")):
        assert discover_ollama_models(timeout=0.1) == []


def test_selected_model_must_remain_installed_before_a_task():
    config = {"execution_mode": "offline", "local_model": "coder:small"}
    assert validate_selected_model(config, "ollama", "coder:small", ["coder:small"])[0]
    valid, error = validate_selected_model(config, "ollama", "coder:small", [])
    assert not valid and "unavailable" in error


def test_offline_menu_restores_two_local_and_two_hosted_choices():
    config = {
        "execution_mode": "offline", "provider": "ollama", "local_model": "qwen2.5-coder:3b",
        "startup_local_model_options": ["qwen2.5-coder:3b", "deepseek-coder:1.3b"],
        "nvidia_model": "nvidia-coder", "models": {"coder": "groq-coder"},
    }
    output = []
    answers = iter(["4", "y"])
    with patch.dict("os.environ", {"GROQ_API_KEY": "test", "NVIDIA_API_KEY": "test"}):
        provider = choose_provider(config, input_fn=lambda prompt: next(answers), output=output.append,
                                   models=["qwen2.5-coder:3b", "deepseek-coder:1.3b"])
    assert provider == "groq"
    assert config["execution_mode"] == "online"
    assert config["selected_model"] == "groq-coder"
    assert all(f"[{index}]" in "\n".join(output) for index in range(1, 5))


def test_offline_to_hosted_requires_explicit_confirmation():
    config = {"execution_mode": "offline", "provider": "ollama", "local_model": "qwen",
              "nvidia_model": "nvidia-coder"}
    answers = iter(["2", "no"])
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}):
        assert choose_provider(config, input_fn=lambda prompt: next(answers), output=lambda value: None,
                               models=["qwen"]) is None
    assert config["execution_mode"] == "offline"
    assert config["provider"] == "ollama"



def test_ollama_discovery_falls_back_to_cli_when_api_returns_empty_list():
    class EmptyResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit): return b'{"models": []}'

    class Completed:
        returncode = 0
        stdout = "NAME ID SIZE MODIFIED\nqwen2.5-coder:3b f72c60 1.9 GB 4 days ago\n"

    with patch("urllib.request.urlopen", return_value=EmptyResponse()):
        with patch("subprocess.run", return_value=Completed()):
            assert discover_ollama_models("http://localhost:11434/v1", timeout=0.1) == [
                "qwen2.5-coder:3b"
            ]


def test_offline_default_uses_discovered_model_even_if_startup_options_are_stale():
    config = {
        "execution_mode": "offline",
        "provider": "ollama",
        "local_model": "qwen2.5-coder:3b",
        "startup_local_model_options": ["deepseek-coder:1.3b"],
        "nvidia_model": "nvidia-coder",
        "models": {"coder": "groq-coder"},
    }
    output = []
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "", "GROQ_API_KEY": ""}):
        provider = choose_provider(
            config,
            input_fn=lambda prompt: "0",
            output=output.append,
            models=["qwen2.5-coder:3b"],
        )
    assert provider == "ollama"
    assert config["selected_model"] == "qwen2.5-coder:3b"
    assert any("qwen2.5-coder:3b" in line for line in output)
