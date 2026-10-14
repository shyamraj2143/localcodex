import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_launcher_resolves_its_own_repository_and_keeps_invocation_workspace(tmp_path):
    powershell = shutil.which("powershell.exe")
    python = shutil.which("py.exe") or shutil.which("python.exe")
    if os.name != "nt" or not powershell or not python:
        pytest.skip("Windows PowerShell and Python launcher are required for the batch launcher smoke test")

    clone = tmp_path / "alternate clone location"
    workspace = tmp_path / "separate workspace"
    clone.mkdir()
    workspace.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "codex.bat", clone / "codex.bat")
    (clone / "main.py").write_text(
        "import json, os, pathlib; print(json.dumps({'script': str(pathlib.Path(__file__).resolve()), 'cwd': os.getcwd()}))",
        encoding="utf-8",
    )
    runner = tmp_path / "invoke-launcher.ps1"
    runner.write_text("param([string]$Launcher)\n& $Launcher\nexit $LASTEXITCODE\n", encoding="utf-8")

    result = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(runner), str(clone / "codex.bat")],
        cwd=workspace, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(output["script"]) == (clone / "main.py").resolve()
    assert Path(output["cwd"]) == workspace.resolve()


def test_launcher_has_no_committed_installation_directory():
    launcher = (Path(__file__).parents[1] / "codex.bat").read_text(encoding="utf-8")
    assert "%~dp0" in launcher
    assert ("F:" + "\\projects\\") not in launcher
    assert ("C:" + "\\GGI\\") not in launcher
