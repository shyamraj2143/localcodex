import json
import os
import tempfile
import unittest

from agent.verification import verify_changed_file
from agent.loop import _is_meaningful_verification_command


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


if __name__ == "__main__":
    unittest.main()
