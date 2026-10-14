import json
import os
import tempfile
import unittest

from agent.verification import verify_changed_file
from agent.loop import _is_meaningful_verification_command, _test_run_executed, _has_passing_behavioral_tests


class ChangedFileVerificationTests(unittest.TestCase):
    def test_valid_python_file_passes_static_check(self):
        with tempfile.TemporaryDirectory() as workspace:
            path = os.path.join(workspace, "main.py")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("print('hello')\n")
            result = verify_changed_file(workspace, "main.py")
            self.assertTrue(result["success"])
            self.assertIn("Python syntax valid", result["checks"])

    def test_invalid_python_file_fails_static_check(self):
        with tempfile.TemporaryDirectory() as workspace:
            with open(os.path.join(workspace, "main.py"), "w", encoding="utf-8") as handle:
                handle.write("def broken(:\n    pass\n")
            result = verify_changed_file(workspace, "main.py")
            self.assertFalse(result["success"])
            self.assertTrue(any("Python syntax error" in error for error in result["errors"]))

    def test_invalid_json_file_fails_static_check(self):
        with tempfile.TemporaryDirectory() as workspace:
            with open(os.path.join(workspace, "config.json"), "w", encoding="utf-8") as handle:
                handle.write('{"enabled": }')
            result = verify_changed_file(workspace, "config.json")
            self.assertFalse(result["success"])
            self.assertTrue(any("JSON syntax error" in error for error in result["errors"]))

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = verify_changed_file(workspace, "../outside.py")
            self.assertFalse(result["success"])

    def test_unrelated_successful_command_is_not_verification(self):
        self.assertFalse(_is_meaningful_verification_command("python -m pip --version"))
        self.assertTrue(_is_meaningful_verification_command("python -m compileall -q ."))
        self.assertTrue(_is_meaningful_verification_command("npm test"))

    def test_syntax_check_alone_is_not_a_successful_behavioral_test(self):
        record = {
            "command": "python -m py_compile main.py",
            "success": True,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        }
        self.assertFalse(_test_run_executed(record))
        self.assertFalse(_has_passing_behavioral_tests([record], ["tests/test_verification.py"]))

    def test_zero_discovered_tests_is_not_success(self):
        record = {
            "command": "python -m pytest -q",
            "success": True,
            "return_code": 0,
            "stdout": "no tests ran",
            "stderr": "",
        }
        self.assertFalse(_test_run_executed(record))
        self.assertFalse(_has_passing_behavioral_tests([record]))

    def test_task_specific_tests_must_be_discovered_and_pass(self):
        passing = {
            "command": "python -m unittest -v test_calculator",
            "success": True,
            "return_code": 0,
            "stdout": "Ran 5 tests in 0.02s\\n\\nOK",
            "stderr": "",
        }
        failing = {
            **passing,
            "success": False,
            "return_code": 1,
            "stdout": "Ran 5 tests in 0.02s",
            "stderr": "FAILED (failures=1)",
        }
        self.assertTrue(_test_run_executed(passing))
        self.assertTrue(_has_passing_behavioral_tests([passing], ["tests/test_calculator.py"]))
        self.assertFalse(_test_run_executed(failing))
        self.assertFalse(_has_passing_behavioral_tests([failing], ["tests/test_calculator.py"]))


if __name__ == "__main__":
    unittest.main()
