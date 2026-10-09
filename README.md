# Local Codex

A terminal-based AI coding assistant that can inspect, create, edit, and debug files in the current workspace. Supports Groq API models and a local Ollama model.

## Requirements

- Python 3.10+
- Git (optional, for version control)
- For recommended NVIDIA mode: an NVIDIA API key from NVIDIA API Catalog
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

4. Configure NVIDIA API (recommended default):

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set `NVIDIA_API_KEY` in `.env`. The default model is `poolside/laguna-xs-2.1` and the endpoint is `https://integrate.api.nvidia.com/v1`. The key stays in your local `.env`; never commit or share it. Groq remains available as option 2.

5. For local Ollama mode, pull one or both supported coding models:

   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ollama pull deepseek-coder:1.3b
   ```

   At startup, option **[1] NVIDIA NIM API** is recommended for stronger coding and instruction following. Choose **[2] Groq API**, **[3] Qwen2.5-Coder 1.5B**, or **[4] DeepSeek-Coder 1.3B** as alternatives. Local models may struggle with complex tasks; the agent can attempt plain-JSON tool calls if native tool calling is unsupported, but this cannot make a small model as capable as a stronger API model.

6. Start Local Codex from the directory you want it to work in:

   ```powershell
   py F:\projects\local-codex\main.py
   ```

   Replace that path with the actual location where you cloned this repository. The current working directory becomes the agent's workspace.

## Usage

Choose Groq or Ollama when prompted, then describe a coding task. Example:

```text
Create a Python calculator in calculator.py with addition, subtraction, multiplication, division, clear, and backspace.
```

Type `/help` for help or `exit` to quit. Use `/task` to enter a multi-line request, then type `END` on its own line to submit the entire task as one prompt.

## Security notes

- Keep API keys in your local `.env` file.
- Do not commit secrets, virtual environments, or Python cache files.
- Review commands and file changes made by an AI coding agent before using them in important projects.
