
import os

from tools.files import (
    list_files,
    read_file,
    write_file,
    edit_file,
)
from tools.search import search_files
from tools.terminal import run_command


def create_folder(workspace, path):
    """Create a folder safely inside the current workspace."""
    try:
        workspace = os.path.realpath(workspace)

        if os.path.isabs(path):
            target = os.path.realpath(path)
        else:
            target = os.path.realpath(
                os.path.join(workspace, path)
            )

        # Prevent creating folders outside the project workspace.
        if os.path.commonpath([workspace, target]) != workspace:
            return {
                "success": False,
                "error": "Folder path must be inside the workspace.",
            }

        os.makedirs(target, exist_ok=True)

        return {
            "success": True,
            "message": "Folder created successfully.",
            "path": os.path.relpath(target, workspace),
            "absolute_path": target,
        }

    except (OSError, ValueError) as error:
        return {
            "success": False,
            "error": str(error),
        }


def get_tools():
    return {
        "list_files": list_files,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "search_files": search_files,
        "run_command": run_command,
        "create_folder": create_folder,
    }


def get_tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and folders in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a project file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a file or replace its complete contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete file contents.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace one unique block in an existing file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for text inside project files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a terminal command in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_folder",
                "description": "Create a new folder inside the project workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Folder path, relative to the workspace or an absolute path inside it.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
    ]
