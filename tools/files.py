import os


def safe_path(workspace, path):

    workspace = os.path.abspath(workspace)
    full_path = os.path.abspath(
        os.path.join(workspace, path)
    )

    if os.path.commonpath(
        [workspace, full_path]
    ) != workspace:
        raise Exception(
            "Access outside project is not allowed."
        )

    return full_path


def list_files(workspace):

    files = []

    ignored = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".idea",
        ".vscode"
    }

    for root, dirs, filenames in os.walk(workspace):

        dirs[:] = [
            d for d in dirs
            if d not in ignored
        ]

        for filename in filenames:

            path = os.path.join(root, filename)

            files.append(
                os.path.relpath(
                    path,
                    workspace
                )
            )

    return files


def read_file(workspace, path):

    full_path = safe_path(workspace, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(path)

    with open(
        full_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


def write_file(workspace, path, content):

    full_path = safe_path(workspace, path)

    directory = os.path.dirname(full_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        full_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content)

    return {
        "success": True,
        "message": f"Created/updated {path}"
    }


def edit_file(
    workspace,
    path,
    old_text,
    new_text
):

    full_path = safe_path(
        workspace,
        path
    )

    if not os.path.exists(full_path):
        raise FileNotFoundError(path)

    with open(
        full_path,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()

    if old_text not in content:

        return {
            "success": False,
            "error": (
                f"Text to replace was not found "
                f"in {path}"
            )
        }

    occurrences = content.count(old_text)

    if occurrences > 1:

        return {
            "success": False,
            "error": (
                f"old_text occurs {occurrences} times. "
                "Provide a larger unique block."
            )
        }

    new_content = content.replace(
        old_text,
        new_text,
        1
    )

    with open(
        full_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(new_content)

    return {
        "success": True,
        "message": f"Edited {path}"
    }