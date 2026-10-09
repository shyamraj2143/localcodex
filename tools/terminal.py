import subprocess


def run_command(
    workspace,
    command
):

    dangerous = [
        "format ",
        "diskpart",
        "shutdown",
        "rmdir /s",
        "del /s /q",
        "rm -rf",
        "sudo"
    ]

    command_lower = command.lower()

    for item in dangerous:

        if item in command_lower:

            return {
                "success": False,
                "error": (
                    f"Blocked dangerous command: {item}"
                )
            }

    try:

        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120
        )

        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "error": "Command timed out after 120 seconds."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }