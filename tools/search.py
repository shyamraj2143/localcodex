import os


def search_files(workspace, query):

    results = []

    ignored = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".idea",
        ".vscode"
    }

    for root, dirs, files in os.walk(workspace):

        dirs[:] = [
            d for d in dirs
            if d not in ignored
        ]

        for filename in files:

            path = os.path.join(
                root,
                filename
            )

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    content = file.read()

                if query.lower() in content.lower():

                    results.append(
                        os.path.relpath(
                            path,
                            workspace
                        )
                    )

            except (
                UnicodeDecodeError,
                PermissionError
            ):
                continue

    return results