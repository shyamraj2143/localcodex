# Local Codex

A terminal-based AI coding assistant that can inspect, create, edit, and debug files in the current workspace. Supports Groq API models and a local Ollama model.

## Requirements

- Python 3.10+
- Git (optional, for version control)
- For Groq mode: a Groq API key
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

4. For Groq mode, copy the example environment file and set your key:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Replace the placeholder with your own key. Never commit `.env` or share your API key.

5. For local Ollama mode, pull the configured model:

   ```powershell
   ollama pull qwen2.5-coder:1.5b
   ```

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

Type `/help` for help or `exit` to quit.

## Security notes

- Keep API keys in your local `.env` file.
- Do not commit secrets, virtual environments, or Python cache files.
- Review commands and file changes made by an AI coding agent before using them in important projects.
