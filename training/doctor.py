"""Best-effort local hardware and Ollama compatibility report."""

from __future__ import annotations

import importlib.util
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _ollama(base_url: str, model: str) -> dict[str, Any]:
    root = base_url.removesuffix("/v1").rstrip("/")
    result: dict[str, Any] = {"base_url": root, "reachable": False, "model_available": False}
    try:
        with urllib.request.urlopen(root + "/api/version", timeout=2) as response:
            result["version"] = json.loads(response.read().decode("utf-8")).get("version")
            result["reachable"] = True
        request = urllib.request.Request(root + "/api/show", data=json.dumps({"name": model}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
        details = data.get("details", {})
        result.update({"model_available": True, "family": details.get("family"), "parameter_size": details.get("parameter_size"), "quantization": details.get("quantization_level"), "format": details.get("format"), "license_metadata_present": bool(data.get("license")), "license_status": "metadata present; terms and training permission not independently reviewed" if data.get("license") else "not reported by model metadata"})
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    return result


def hardware_report(root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    packages = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "pytest")}
    gpu = {"nvidia_smi": shutil.which("nvidia-smi"), "cuda_available": None, "devices": []}
    if packages["torch"]:
        try:
            import torch  # type: ignore[import-not-found]
            gpu["cuda_available"] = bool(torch.cuda.is_available())
            if gpu["cuda_available"]:
                gpu["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        except Exception as exc:
            gpu["torch_error"] = str(exc)
    model = str(config.get("training", {}).get("base_model", config.get("local_model", "qwen2.5-coder:3b")))
    url = str(config.get("ollama_base_url", "http://localhost:11434/v1"))
    ollama = _ollama(url, model)
    estimates = {"full_finetune_3b_fp16_minimum_gib": 30, "qlora_3b_practical_vram_gib": 6, "note": "Rough planning estimates; actual memory depends on sequence length, optimizer, batch size, and framework."}
    supported = bool(packages["torch"] and packages["transformers"] and packages["trl"] and gpu["cuda_available"])
    limitations = []
    if not ollama["reachable"]: limitations.append("Ollama endpoint is unreachable")
    if ollama["model_available"] and ollama.get("format", "").lower() == "gguf": limitations.append("Ollama model package is an inference artifact, not a verified fine-tuning base checkpoint")
    if not packages["transformers"] or not packages["trl"]: limitations.append("Transformers/TRL training dependencies are missing")
    if not gpu["cuda_available"]: limitations.append("CUDA GPU not confirmed; practical QLoRA is unsupported by the detected environment")
    ram_bytes = None
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram_bytes = status.ullTotalPhys
        except (AttributeError, OSError):
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "system_ram_bytes": ram_bytes, "disk_free_bytes": disk.free, "gpu": gpu, "training_packages": packages, "ollama": ollama, "model_identity": model, "training_compatible": supported, "estimated_requirements": estimates, "recommended_strategy": "Prepare/evaluate offline; install a compatible Transformers source checkpoint and use QLoRA on a supported CUDA GPU" if not supported else "Begin with a small QLoRA smoke run after verifying source checkpoint license and tokenizer", "limitations": limitations}
