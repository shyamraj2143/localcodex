import os
import tempfile
import unittest

from tools.files import edit_file, read_file, safe_path, write_file
from tools.registry import create_folder


class WorkspaceToolTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = self.tempdir.name

    def tearDown(self):
        self.tempdir.cleanup()

    def test_write_read_nested_file(self):
        result = write_file(self.workspace, "src/main.py", "print('hello')\n")
        self.assertTrue(result["success"])
        self.assertEqual(read_file(self.workspace, "src/main.py"), "print('hello')\n")

    def test_edit_requires_unique_match(self):
        write_file(self.workspace, "example.txt", "alpha beta alpha")
        result = edit_file(self.workspace, "example.txt", "alpha", "gamma")
        self.assertFalse(result["success"])
        self.assertIn("occurs 2 times", result["error"])

    def test_rejects_parent_traversal(self):
        with self.assertRaises(ValueError):
            safe_path(self.workspace, "../outside.txt")

    def test_create_folder_inside_workspace(self):
        result = create_folder(self.workspace, "calculator_gui")
        self.assertTrue(result["success"])
        self.assertTrue(os.path.isdir(os.path.join(self.workspace, "calculator_gui")))

    def test_rejects_folder_outside_workspace(self):
        parent = os.path.dirname(self.workspace)
        outside = os.path.join(parent, "should_not_be_created_by_test")
        result = create_folder(self.workspace, outside)
        self.assertFalse(result["success"])
        self.assertFalse(os.path.exists(outside))


if __name__ == "__main__":
    unittest.main()
