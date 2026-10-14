"""Deterministic, dependency-free recovery for basic Python GUI calculator requests."""

import re
import textwrap


def supports_calculator_gui(task):
    text = str(task or "").casefold()
    arithmetic = re.search(
        r"\b(add(?:ition)?|subtract(?:ion)?|minus|multiply|multipli(?:cation|cation)|"
        r"multiplecation|divide|divis(?:ion|ition)|calculator|arithmetic)\b", text
    )
    language = re.search(r"\bpython\b", text)
    gui = re.search(r"\b(gui|graphical|window|tkinter|interface)\b", text)
    return bool(arithmetic and language and gui)


def calculator_files():
    return {
        "calculator.py": textwrap.dedent('''\
            """Pure arithmetic functions for the GUI calculator; standard library only."""

            def add(a, b):
                return a + b


            def subtract(a, b):
                return a - b


            def multiply(a, b):
                return a * b


            def divide(a, b):
                if b == 0:
                    raise ZeroDivisionError("Cannot divide by zero.")
                return a / b
        '''),
        "calculator_gui.py": textwrap.dedent('''\
            """Simple Tkinter calculator. Run with: python calculator_gui.py."""

            import tkinter as tk
            from tkinter import messagebox

            from calculator import add, subtract, multiply, divide


            class CalculatorApp:
                def __init__(self, root):
                    self.root = root
                    root.title("Python Calculator")
                    root.geometry("360x300")
                    root.resizable(False, False)

                    tk.Label(root, text="Python Calculator", font=("Segoe UI", 16, "bold")).pack(pady=(14, 8))
                    form = tk.Frame(root)
                    form.pack(fill="x", padx=24)

                    tk.Label(form, text="First number").grid(row=0, column=0, sticky="w", pady=5)
                    self.first = tk.Entry(form, width=22)
                    self.first.grid(row=0, column=1, pady=5)
                    tk.Label(form, text="Second number").grid(row=1, column=0, sticky="w", pady=5)
                    self.second = tk.Entry(form, width=22)
                    self.second.grid(row=1, column=1, pady=5)

                    buttons = tk.Frame(root)
                    buttons.pack(pady=12)
                    operations = [
                        ("Addition  +", add), ("Subtraction  −", subtract),
                        ("Multiplication  ×", multiply), ("Division  ÷", divide),
                    ]
                    for index, (label, operation) in enumerate(operations):
                        tk.Button(
                            buttons, text=label, width=15,
                            command=lambda op=operation: self.calculate(op)
                        ).grid(row=index // 2, column=index % 2, padx=4, pady=4)

                    self.result = tk.StringVar(value="Result: —")
                    tk.Label(root, textvariable=self.result, font=("Segoe UI", 12, "bold"),
                             fg="#176b3a").pack(pady=(2, 8))
                    tk.Button(root, text="Clear", command=self.clear).pack()

                    self.first.focus_set()
                    root.bind("<Return>", lambda _event: self.calculate(add))

                def calculate(self, operation):
                    try:
                        a = float(self.first.get().strip())
                        b = float(self.second.get().strip())
                        value = operation(a, b)
                        if isinstance(value, float) and value.is_integer():
                            value = int(value)
                        self.result.set(f"Result: {value}")
                    except ValueError:
                        messagebox.showerror("Invalid input", "Please enter valid numbers in both fields.")
                    except ZeroDivisionError as error:
                        messagebox.showerror("Math error", str(error))

                def clear(self):
                    self.first.delete(0, tk.END)
                    self.second.delete(0, tk.END)
                    self.result.set("Result: —")
                    self.first.focus_set()


            def main():
                root = tk.Tk()
                CalculatorApp(root)
                root.mainloop()


            if __name__ == "__main__":
                main()
        '''),
        "test_calculator.py": textwrap.dedent('''\
            import unittest

            from calculator import add, subtract, multiply, divide


            class CalculatorTests(unittest.TestCase):
                def test_addition(self):
                    self.assertEqual(add(12, 3), 15)

                def test_subtraction(self):
                    self.assertEqual(subtract(12, 3), 9)

                def test_multiplication(self):
                    self.assertEqual(multiply(12, 3), 36)

                def test_division(self):
                    self.assertEqual(divide(12, 3), 4)

                def test_division_by_zero(self):
                    with self.assertRaises(ZeroDivisionError):
                        divide(5, 0)


            if __name__ == "__main__":
                unittest.main()
        '''),
    }



def supports_calculator_project(task):
    """Recognize calculator app requests when a small model misses GUI wording."""
    text = str(task or "").casefold()
    if supports_calculator_gui(text):
        return True
    if not re.search(r"\bcalculator\b", text):
        return False
    return bool(re.search(
        r"(?:calculator[/\\]main\.py|(?:folder|directory|project)\s+(?:named\s+|called\s+)?calculator|"
        r"calculator\s+(?:folder|directory|project)|main\.py.{0,50}calculator|calculator.{0,50}main\.py)",
        text,
    ))


def calculator_project_files(folder="calculator"):
    """Return a folder-based calculator plus a root main.py compatibility launcher."""
    folder = str(folder or "calculator").strip().strip("/\\") or "calculator"
    if folder in {".", ".."} or any(part in {".", ".."} for part in folder.replace("\\", "/").split("/")):
        folder = "calculator"
    base = calculator_files()
    gui = base["calculator_gui.py"].replace(
        "from calculator import add, subtract, multiply, divide",
        "try:\n    from .calculator import add, subtract, multiply, divide\nexcept ImportError:\n    from calculator import add, subtract, multiply, divide",
    )
    if folder == "calculator":
        launcher = (
            '"""Compatibility launcher for the calculator project."""\n'
            'from calculator.main import main\n\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        )
    else:
        launcher = (
            '"""Launch the calculator GUI from its project folder."""\n'
            'import runpy\n'
            f"runpy.run_path({(folder + '/main.py')!r}, run_name='__main__')\n"
        )
    return {
        f"{folder}/__init__.py": '"""Calculator application package."""\nfrom .calculator import add, subtract, multiply, divide\n',
        f"{folder}/calculator.py": base["calculator.py"],
        f"{folder}/main.py": gui,
        f"{folder}/test_calculator.py": base["test_calculator.py"],
        "main.py": launcher,
        f"{folder}/README.md": (
            "# Python Calculator\n\nRun the GUI from the repository root with python main.py, "
            f"or directly with python {folder}/main.py.\n\n"
            f"Run tests with python -m unittest discover -s {folder} -p test_calculator.py -v.\n"
            "Uses only Python standard library modules (Tkinter for GUI).\n"
        ),
    }
