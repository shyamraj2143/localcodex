import unittest

from agent.quick_apps import detect_quick_app, quick_app_files


class QuickAppTests(unittest.TestCase):
    def test_detects_the_four_short_prompts(self):
        self.assertEqual(detect_quick_app("Create a calculator with a GUI."), "calculator")
        self.assertEqual(detect_quick_app("Create a number guessing game with a GUI."), "number_guessing_game")
        self.assertEqual(detect_quick_app("Create a to-do list app with a GUI."), "todo_list_app")
        self.assertEqual(detect_quick_app("Create an expense tracker with a GUI."), "expense_tracker")

    def test_unrelated_prompt_is_not_routed_to_demo_templates(self):
        self.assertIsNone(detect_quick_app("Explain binary search in Python."))

    def test_each_non_calculator_app_has_entrypoint_logic_and_tests(self):
        for app in ("number_guessing_game", "todo_list_app", "expense_tracker"):
            with self.subTest(app=app):
                files = quick_app_files(app)
                self.assertIn(app + "/main.py", files)
                self.assertIn(app + "/app_logic.py", files)
                self.assertIn(app + "/test_app_logic.py", files)
                self.assertIn("main.py", files)
                self.assertIn("import tkinter", files[app + "/main.py"])


if __name__ == "__main__":
    unittest.main()
