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

2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install Python dependencies:

   ```powershell
   py -m pip install -r requirements.txt
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

   The checked-in config uses `execution_mode: "offline"`, provider `ollama`, and default model `qwen2.5-coder:3b`. Start with `py main.py` from the folder you want the agent to modify. Startup lists installed local models. To use a hosted provider, change `execution_mode` to `online` in `config/config.json` and set its key in the project `.env`. `.env` is loaded from the repository even when launched in another workspace, and the current directory remains the task workspace. Local models must be installed; before each task the CLI checks that the selected model is available. `auto` uses the configured provider without silent provider switching.

   Local mode uses plain JSON actions, readable tool feedback, duplicate-action protection, missing-deliverable checks, and retries when the model tries to finish too early. If the model repeats the same tool action three times while requested files are still missing, Local Codex switches to a constrained no-tools file-generation recovery, accepts only explicitly requested file paths, writes the files itself, and runs the most relevant available check. Hinglish folder/file requests like asking to create codex_smoke_test and put main.py inside it are included in missing-deliverable detection. After every successful source-file write/edit, Local Codex reads the file back and performs supported static checks (Python syntax, JSON syntax, and JavaScript syntax when Node.js is installed). It also records executed commands and only counts recognized test/build/lint/syntax commands as validation; a command such as `python -m pip --version` is not considered a code test. The final response includes changed files, automatic checks, commands and return codes, and an honest verification status. Static syntax checks do not replace relevant tests or real end-to-end testing, and a small model can still have capability limits.

   Coding sessions use the event-driven student-demo dashboard. It shows the request, current phase, real workflow events, actual parallel read workers, changed files, executed checks and honest verification state. Internal step counters are omitted. Set `dashboard_demo_mode` to `false` to disable beginner-friendly phrasing; set `dashboard_language` to `"hinglish"` for Hinglish activity descriptions; set `dashboard_show_technical` to `true` for a compact technical event log. `ui_mode` remains `auto` (Rich on an interactive terminal, plain text otherwise), `rich`, or `plain`; non-interactive output uses readable plain text. Independent read-only tool calls can run concurrently; file edits and commands stay ordered. `max_parallel_tool_workers` bounds ordinary read operations (default 2); `max_concurrent_llm_requests` bounds model calls (default 1). A single Ollama generation is not made faster by adding threads. Task, model-request and tool-operation durations are recorded in the final report.

   Before each planning or coding request, Local Codex builds a compact context from the active workspace: ignored-file-aware inventory, manifests, README, entry points, test files, relevant source excerpts and Python definitions. It refreshes that context after edits and commands, carries the previous execution report into follow-up debugging, and routes explicit explanation-only requests through a no-tools path. Start Local Codex from the project directory you want it to inspect.

   Install development test tools and run the full suite:

   ```powershell
   py -m pip install -r requirements-dev.txt
   py -m pytest -q
   ```

   An optional local latency comparison is available with `py -m tools.benchmark_ollama`. It runs two equivalent sequential requests against the configured installed Ollama model. Add `--include-parallel` to opt into two concurrent generations; this can exhaust RAM on an 8 GB CPU-only machine. The benchmark measures response latency and failures only, not coding completion or test quality. No benchmark is run automatically.

6. Start Local Codex from the directory you want it to work in:

   ```powershell
   py F:\projects\local-codex\main.py
   ```

   Replace that path with the actual location where you cloned this repository. The current working directory becomes the agent's workspace.

## Usage

Start from PowerShell with `py F:\projects\local-codex\main.py` (or run `python main.py` from the repository). Startup lists the four configured provider/model choices and lets you reuse the local default. NVIDIA/Groq require explicit confirmation before leaving offline mode. Example:

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
