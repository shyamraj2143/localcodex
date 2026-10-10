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

   Set `NVIDIA_API_KEY` in `.env`. The default model is `qwen/qwen2.5-coder-32b-instruct` and the endpoint is `https://integrate.api.nvidia.com/v1`. The key stays in your local `.env`; never commit or share it. Groq remains available as option 2.

5. For local Ollama mode, pull a supported coding model:

   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ollama pull deepseek-coder:1.3b
   ```

   For better task understanding and multi-file coding, use a stronger local model if your RAM/VRAM allows it:

   ```powershell
   ollama pull qwen2.5-coder:3b
   ```

   The checked-in config defaults the local model to `qwen2.5-coder:3b`. Option **[3] Qwen Coder** respects this value instead of silently forcing 1.5B; option **[4]** selects DeepSeek-Coder 1.3B. The checked-in provider default is now **Ollama**, so press Enter to use Qwen Coder 3B locally. Choose **[1]** for NVIDIA or **[2]** for Groq.

   Local mode uses plain JSON actions, readable tool feedback, duplicate-action protection, missing-deliverable checks, and retries when the model tries to finish too early. If the model repeats the same tool action three times while requested files are still missing, Local Codex switches to a constrained no-tools file-generation recovery, accepts only explicitly requested file paths, writes the files itself, and runs the most relevant available check. Hinglish folder/file requests like asking to create codex_smoke_test and put main.py inside it are included in missing-deliverable detection. After every successful source-file write/edit, Local Codex reads the file back and performs supported static checks (Python syntax, JSON syntax, and JavaScript syntax when Node.js is installed). It also records executed commands and only counts recognized test/build/lint/syntax commands as validation; a command such as `python -m pip --version` is not considered a code test. The final response includes changed files, automatic checks, commands and return codes, and an honest verification status. Static syntax checks do not replace relevant tests or real end-to-end testing, and a small model can still have capability limits.

   Run the project tests after updating:

   ```powershell
   py -m unittest discover -s tests -v
   ```

6. Start Local Codex from the directory you want it to work in:

   ```powershell
   py F:\projects\local-codex\main.py
   ```

   Replace that path with the actual location where you cloned this repository. The current working directory becomes the agent's workspace.

## Usage

Choose NVIDIA, Groq, or Ollama when prompted, then describe a coding task. Example:

```text
Create a Python calculator in calculator.py with addition, subtraction, multiplication, division, clear, and backspace.
```

Type `/help` for help or `exit` to quit. Use `/task` to enter a multi-line request, then type `END` on its own line to submit the entire task as one prompt.

## Security notes

- Keep API keys in your local `.env` file.
- Do not commit secrets, virtual environments, or Python cache files.
- Review commands and file changes made by an AI coding agent before using them in important projects.
