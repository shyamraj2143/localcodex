import unittest

from agent.calculator_recovery import calculator_files, supports_calculator_gui


class CalculatorRecoveryTests(unittest.TestCase):
    def test_accepts_user_request_with_common_typos(self):
        self.assertTrue(
            supports_calculator_gui(
                "craete addition subtraction multiplecation divisition in python with gui"
            )
        )

    def test_rejects_non_gui_request(self):
        self.assertFalse(supports_calculator_gui("write python addition subtraction functions"))

    def test_produces_gui_logic_and_tests_without_external_packages(self):
        files = calculator_files()
        self.assertEqual(set(files), {"calculator.py", "calculator_gui.py", "test_calculator.py"})
        self.assertIn("import tkinter", files["calculator_gui.py"])
        self.assertNotIn("import pygame", files["calculator_gui.py"])
        self.assertIn("ZeroDivisionError", files["calculator.py"])
        self.assertIn("test_division_by_zero", files["test_calculator.py"])


if __name__ == "__main__":
    unittest.main()
