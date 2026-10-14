import sys

from tools.terminal import run_command


def _python_command(code):
    return f'"{sys.executable}" -c "{code}"'


def test_command_result_contains_cwd_duration_output_and_exit_code(tmp_path):
    result = run_command(str(tmp_path), _python_command("print('hello')"))
    assert result["success"] is True
    assert result["return_code"] == 0
    assert result["stdout"].strip() == "hello"
    assert result["cwd"] == str(tmp_path.resolve())
    assert result["duration_seconds"] >= 0
    assert result["timed_out"] is False


def test_command_failure_and_stderr_are_preserved(tmp_path):
    result = run_command(str(tmp_path), _python_command("import sys; print('broken', file=sys.stderr); sys.exit(7)"))
    assert result["success"] is False
    assert result["return_code"] == 7
    assert "broken" in result["stderr"]


def test_blocked_command_has_a_structured_outcome(tmp_path):
    result = run_command(str(tmp_path), "shutdown /s")
    assert result["success"] is False
    assert result["blocked"] is True
    assert result["command"] == "shutdown /s"
    assert result["cwd"] == str(tmp_path.resolve())


def test_interactive_command_receives_eof_instead_of_waiting_for_terminal_input(tmp_path):
    result = run_command(
        str(tmp_path),
        _python_command("input('Enter char to delete up to: ')"),
        timeout=2,
    )
    assert result["success"] is False
    assert result["timed_out"] is False
    assert result["duration_seconds"] < 2
    assert "EOFError" in result["stderr"]


def test_timed_out_command_is_terminated(tmp_path):
    result = run_command(
        str(tmp_path),
        _python_command("import time; time.sleep(10)"),
        timeout=0.2,
    )
    assert result["success"] is False
    assert result["timed_out"] is True
    assert "terminated" in result["error"]
    assert result["duration_seconds"] < 5
