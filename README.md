# Local Codex

A terminal-based AI coding assistant that can inspect, create, edit, and debug files in the current workspace. Supports NVIDIA NIM, Groq API models, and local Ollama models.

## Requirements

- Python 3.10+
- Git (optional, for version control)
- For optional NVIDIA mode: an NVIDIA API key from NVIDIA API Catalog
- For Groq mode: a Groq API key (optional fallback)
- For local mode: Ollama installed and a model pulled locally

## Setup (Windows PowerShell)

1. Clone the repository and enter the folder:

   ```powershell
   git clone https://github.com/shyamraj2143/localcodex.git
   cd localcodex
   ```

2. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

3. Install Python dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

4. Configure NVIDIA API (optional, for online mode):

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set `NVIDIA_API_KEY` and/or `GROQ_API_KEY` in `.env` for hosted providers. The configured NVIDIA model is `openai/gpt-oss-20b` at `https://integrate.api.nvidia.com/v1`. The key stays in your local `.env`; never commit or share it.

5. For local Ollama mode, pull a supported coding model:

   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ollama pull deepseek-coder:1.3b
   ```

   For better task understanding and multi-file coding, use a stronger local model if your RAM/VRAM allows it:

   ```powershell
   ollama pull qwen2.5-coder:3b
   ```

   The checked-in config uses `execution_mode: "hybrid"` and displays exactly four coding-model choices at startup: (1) NVIDIA Qwen2.5-Coder 32B, (2) Groq Qwen3 32B, (3) local DeepSeek Coder 1.3B, and (4) local Qwen2.5-Coder 3B. Hosted choices require their respective API keys; local choices require the exact model tag to be installed. For local choices, Groq can turn complex requests into a structured implementation plan and, if local coding fails verification, may perform one bounded hosted coding/repair attempt. If `GROQ_API_KEY` is missing or unreachable, the architect falls back to a local task specification and hosted repair is unavailable. The Groq architect does not receive permission to run workspace tools. Because request understanding sends the user task and compact workspace context to Groq, use this mode only if you are comfortable sending that context to the hosted API. Start with the repository launcher (or the `codex` command setup below) from the folder you want the agent to modify. `.env` is loaded from the repository even when launched in another workspace, and the current directory remains the task workspace. Local models must be installed; before each task the CLI checks that the selected model is available. `offline` blocks hosted requests, `online` allows provider selection, and `hybrid` shows the four fixed model choices and enables local coding with optional hosted assistance.

   Optional local overrides may be placed in the untracked `.env` file:
   `OLLAMA_HOST=http://localhost:11434` and
   `OLLAMA_MODEL=qwen2.5-coder:3b`. If unset, the checked-in config defaults
   are used. Model discovery checks the Ollama tags API and falls back to
   `ollama list` when the API returns an empty model list. It never downloads
   a model automatically.

   Local mode uses plain JSON actions, readable tool feedback, duplicate-action protection, missing-deliverable checks, and retries when the model tries to finish too early. If the model repeats the same tool action three times while requested files are still missing, Local Codex switches to a constrained no-tools file-generation recovery, accepts only explicitly requested file paths, writes the files itself, and runs the most relevant available check. Hinglish folder/file requests like asking to create codex_smoke_test and put main.py inside it are included in missing-deliverable detection. After every successful source-file write/edit, Local Codex reads the file back and performs supported static checks (Python syntax, JSON syntax, and JavaScript syntax when Node.js is installed). It also records executed commands and only counts recognized test/build/lint/syntax commands as validation; a command such as `python -m pip --version` is not considered a code test. The final response includes changed files, automatic checks, commands and return codes, and an honest verification status. Static syntax checks do not replace relevant tests or real end-to-end testing, and a small model can still have capability limits.

   Coding sessions use a compact event-driven terminal dashboard: provider/model/workspace header, one live status line, a short build checklist, and a concise result. Status and checklist changes come from real tool, file, test, and task events; no fixed percentages or worker cards are shown. Set `dashboard_demo_mode` to `false` to disable beginner-friendly phrasing; set `dashboard_language` to `"english"` for English status text; set `dashboard_show_technical` to `true` for a compact technical event log. `ui_mode` remains `auto` (Rich on an interactive terminal, plain text otherwise), `rich`, or `plain`; non-interactive output uses readable plain text. Independent read-only tool calls can run concurrently; file edits and commands stay ordered. `max_parallel_tool_workers` bounds ordinary read operations (default 2); `max_concurrent_llm_requests` bounds concurrent model calls (default 1); `max_total_workers` caps the combined configured worker pools (default 3). A single Ollama generation is not made faster by adding threads. Task, model-request and tool-operation durations are recorded in the final report. The default combined worker-pool cap is 3, local model concurrency stays at 1, the task deadline is 300 seconds, and workspace context is bounded to 6,500 characters plus 1,200 characters of memory; adjust `max_total_workers`, `task_deadline_seconds`, `workspace_context_chars`, and `workspace_memory_context_chars` in `config/config.json` as needed.

   Before each planning or coding request, Local Codex builds a compact context from the active workspace: ignored-file-aware inventory, manifests, README, entry points, test files, relevant source excerpts and Python definitions. It caches unchanged context and refreshes it after edits and commands, carries the previous execution report into follow-up debugging, and routes explicit explanation-only requests through a no-tools path. Start Local Codex from the project directory you want it to inspect.

   Install development test tools and run the full suite:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
   .\.venv\Scripts\python.exe -m pytest -q
   ```

   An optional local latency comparison is available with `.\.venv\Scripts\python.exe -m tools.benchmark_ollama`. It runs two equivalent sequential requests against the configured installed Ollama model. Add `--include-parallel` to opt into two concurrent generations; this can exhaust RAM on an 8 GB CPU-only machine. The benchmark measures response latency and failures only, not coding completion or test quality. No benchmark is run automatically.

6. Start Local Codex while keeping the current directory as the workspace:

   ```powershell
   $codexRepo = (Resolve-Path "$env:USERPROFILE\src\localcodex").Path
   python "$codexRepo\main.py"
   ```

   Replace the example clone directory with your actual repository location. The portable launcher uses the repository's `.venv` Python directly when it exists; activation is not required.

### Open Local Codex by typing `codex` in the terminal

The repository includes a portable `codex.bat`. To make `codex` available globally without committing a machine-specific path, create a user-local shim pointing to your clone:

1. In PowerShell, set the actual directory where you cloned this repository:

   ```powershell
   $repo = (Resolve-Path "C:\path\to\your\localcodex").Path
   $bin = Join-Path $env:USERPROFILE "bin"
   New-Item -ItemType Directory -Force -Path $bin | Out-Null
   "@echo off`r`ncall `"$repo\codex.bat`" %*`r`n" | Set-Content -Encoding ascii (Join-Path $bin "codex.cmd")
   ```

2. Add `$bin` (normally `%USERPROFILE%\bin`) to your **user Path** using Windows **Edit environment variables for your account** > **Path** > **New**. This adds one entry and leaves existing entries untouched.
3. Close and reopen Command Prompt or PowerShell.
4. Change to the project you want Codex to work on, then run:

   ```powershell
   cd C:\path\to\your\project
   codex
   ```

The shim calls the clone's launcher. It resolves `main.py` relative to the launcher, uses the repository `.venv` Python when available, and otherwise tries the Windows `py -3` launcher or `python` on `PATH`. The current folder becomes the coding workspace. Optionally set `LOCAL_CODEX_WORKSPACE` to an explicit workspace directory before launching. Ollama and the configured model must be installed for the default offline mode.

## Usage

Start from PowerShell with `python "<clone-directory>\main.py"` or type `codex` after installing the user-local shim above. Startup lists the four configured provider/model choices and lets you reuse the local default. NVIDIA/Groq require explicit confirmation before leaving offline mode. Example:

```text
Create a Python calculator in calculator.py with addition, subtraction, multiplication, division, clear, and backspace.
```

Type `/help` for help or `exit` to quit. Use `/task` to enter a multi-line request, then type `END` on its own line to submit the entire task as one prompt.

## Persistent workspace memory

On startup, Local Codex indexes the selected workspace and stores compact metadata
under `.localcodex/memory/`: a project summary, file index, inferred import
relationships, discovered and executed commands, task reports, unresolved issues,
and bounded training trajectories. Current files remain the source of truth;
memory is reconciled at startup and refreshed around model turns. Secret files,
virtual environments, generated output, and `.localcodex/` itself are excluded.
The terminal dashboard shows the actual index and retrieval counts.

Memory is enabled by `persistent_memory_enabled` in `config/config.json`; set it
to `false` to disable persistence. `workspace_memory_dir` can select another
relative directory inside the workspace. `.localcodex/` is ignored by Git by
default; remove that ignore rule if project memory should be version-controlled.
Trajectory records are compact, secret-redacted, and verification-labeled; they
are not automatically treated as positive training examples.

To curate stored task records, export verified examples separately from failed
diagnostics:

```powershell
python -m training.cli trajectories export --input .\path\to\workspace\.localcodex\memory\trajectories.jsonl
```

Only `PASS` records become training examples. Other trajectories are written to
`diagnostics.jsonl` with failure labels for review.

## Task planning and provider routing

Complex build requests are classified before coding. In `online` or `auto` mode, when hosted access is
enabled and a real key plus configured model are available, Local Codex asks
the configured Groq or NVIDIA model for a JSON implementation contract. The
contract is schema-checked, workspace-relative paths are validated, and the
local coding agent receives it alongside the original task. If hosted planning
is unavailable or returns invalid JSON, a deterministic local contract is used.
Simple requests skip the architect. After an incomplete local attempt, the
agent can make one bounded hosted-coder repair attempt using sanitized task
context, changed files, and actual command results. Offline mode disables both
hosted planning and repair escalation, and blocks common package installation,
download, web, and remote-clone commands in the terminal tool. Use dependencies
already present on the machine; the agent never pulls a missing Ollama model.

`execution_mode` accepts `offline`, `online`, or `auto`. Offline forces Ollama
and denies hosted client construction even if another provider is requested.
Online keeps interactive provider and model selection for configured/credentialed providers.
Auto uses `provider`; it does not silently switch providers after an error.
Router settings in
`config/config.json` include `architect_provider`, `hosted_provider_preference`,
`provider_fallback_enabled`, `hosted_api_enabled`, `local_only`,
`hosted_coder_escalation`, `escalation_provider`, `max_escalations`, and
`request_timeout_seconds`, `nvidia_request_timeout_seconds`, and
`retry.max_attempts` / `retry.base_delay`. `provider_fallback_enabled` is retained for older
configuration files but provider failover is disabled to preserve the selected
provider. Set `local_only` to `true` or
`hosted_api_enabled` to `false` to prevent hosted calls. API keys are read from
the existing project `.env`; placeholder values do not count as credentials.
No key values are included in prompts or reports. Tool output is redacted, and
direct reads of `.env`, private-key, and credential files are blocked.

The configured model IDs are Groq roles in `models`, NVIDIA
`openai/gpt-oss-20b`, and local Ollama `qwen2.5-coder:3b`. The NVIDIA model
was verified with live text and registered-tool requests. Provider availability
and account access can still change. Provider errors are reported without
switching providers.

If a model fails, the CLI asks whether to retry that provider, choose another
configured provider/model, or explicitly continue with an installed local
Ollama model. Offline mode blocks hosted choices. After a provider failure,
the specifically supported Python car-racing GUI request can use a bounded
Tkinter implementation through the workspace file tools; other unsupported
tasks remain failed until the user chooses a recovery action.

Completion checks include required deliverables, non-empty contents, static
file checks, and real test discovery where the task contract requires tests.
Game tasks additionally reject Hello World/placeholder output and require
detectable gameplay loop and input handling. GUI visual inspection is not
automated; game logic tests and execution records are used as evidence.

Run routing and completeness tests with:

```powershell
py -m pytest -q
```

Hosted API calls are mocked in tests. Live provider calls are not part of the
test suite and may incur usage charges when used interactively. Parallel work
currently covers independent read-only tool operations; it does not run
independent coding subtasks through multiple LLM workers. Ollama LLM concurrency
defaults to one request for the 8 GB / CPU-only setup.

## Security notes

- Keep API keys in your local `.env` file.
- Do not commit secrets, virtual environments, or Python cache files.
- Review commands and file changes made by an AI coding agent before using them in important projects.
# Local training data tools

Local Codex includes an offline-first training-data preparation slice. It does
not download datasets or models. Keep input datasets and generated artifacts
outside your coding workspace; the default output is the ignored
`training_data/` directory in this application repository.

```powershell
python -m training.cli doctor
python -m training.cli datasets validate --input .\my-data.jsonl
python -m training.cli datasets prepare --input .\my-data.jsonl --output .\training_data\prepared
python -m training.cli datasets stats --input .\training_data\prepared\train.jsonl
python -m training.cli train --dry-run
```

JSON, JSONL/NDJSON, and CSV records can use `messages` with system/user/
assistant roles, or `prompt`/`response`, `instruction`/`output`, or
`task`/`solution` fields. Parquet requires optional pandas and pyarrow. The
preparer rejects malformed/empty rows, redacts common credential patterns,
deduplicates normalized conversations, preserves metadata, and writes a
manifest plus reproducible train/validation/test splits. Split leakage is
prevented for records with identical content hashes. Inspect the manifest and
verify dataset license and model-training rights before any training.

`doctor` reports the configured Ollama model metadata and local training
dependencies. Ollama package weights are treated as inference artifacts, not
as verified fine-tuning checkpoints. `train --dry-run` only reports resource
estimates; actual weight training is not available in this release. It exits
with an explicit error if invoked without `--dry-run`. No dataset is collected
unless you supply a local input file. Trajectory recording is not enabled.

Run the standard-library training data tests with
`python -m unittest tests.test_training_data -v`. The existing test suite uses
pytest; install the project's test dependency in a compatible environment to
run it (pytest is not installed in the checked-in virtual environment).


## Troubleshooting Windows and VS Code

- If `codex` is not recognized, verify that the folder containing `codex.bat`
  is discoverable through the user PATH, then open a new terminal. The portable
  repository launcher itself uses paths relative to its own location; a
  machine-specific global `codex.cmd` shim belongs in the user's home folder,
  not in the repository.
- If the installed Ollama model is reported missing, run `ollama list`, check
  `OLLAMA_HOST` and `OLLAMA_MODEL` in the local `.env`, then restart Codex.
  Never commit the real `.env` file.
- If CMD displays an unexpected prompt such as `Enter char to delete up to:`,
  press Ctrl+C once to stop the current interactive command. Open a new CMD
  window and check whether the prompt recurs before running Codex. Do not delete
  registry entries or startup scripts without identifying the source.
- If Ctrl+clicking files in VS Code opens a blank editor/window showing only the
  VS Code logo, this is usually a local VS Code/editor-state, extension, or
  keybinding issue rather than a Python agent setting. First try opening the
  file with a normal click, use `File > Open File...`, and run
  `Developer: Reload Window` from the Command Palette. If it persists, test
  with extensions disabled using `code --disable-extensions`. Repository
  changes cannot repair a local VS Code profile or a global extension setting.
