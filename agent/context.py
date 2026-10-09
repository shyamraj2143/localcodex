from tools.files import list_files


def build_project_context(workspace):

    files = list_files(workspace)

    if not files:
        return "Workspace is empty."

    result = "PROJECT FILES:\n\n"

    for file in files:
        result += f"- {file}\n"

    return result