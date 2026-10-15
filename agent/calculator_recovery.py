"""Deterministic, dependency-free recovery for Python GUI calculator requests."""

import ast
import operator
import re
import textwrap


def supports_calculator_gui(task):
    text = str(task or "").casefold()
    arithmetic = re.search(
        r"\b(add(?:ition)?|subtract(?:ion)?|minus|multiply|multipli(?:cation|cation)|"
        r"multiplecation|divide|divis(?:ion|ition)|calculator|arithmetic)\b", text
    )
    language = re.search(r"\bpython\b", text)
    gui = re.search(r"\b(gui|graphical|window|tkinter|interface|keypad)\b", text)
    return bool(arithmetic and language and gui)


def calculator_files():
    return {
        "calculator.py": textwrap.dedent('''\\
            """Safe arithmetic expression evaluator; no third-party dependencies."""
            import ast
            import operator

            _BINARY = {
                ast.Add: operator.add, ast.Sub: operator.sub,
                ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            _UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

            def evaluate(expression):
                """Evaluate numeric arithmetic only; never use eval()."""
                try:
                    tree = ast.parse(str(expression), mode="eval")
                except (SyntaxError, ValueError) as exc:
                    raise ValueError("Enter a valid arithmetic expression.") from exc

                def visit(node):
                    if isinstance(node, ast.Expression):
                        return visit(node.body)
                    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                        return node.value
                    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                        left, right = visit(node.left), visit(node.right)
                        if isinstance(node.op, ast.Pow) and abs(right) > 100:
                            raise ValueError("Exponent is too large.")
                        return _BINARY[type(node.op)](left, right)
                    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
                        return _UNARY[type(node.op)](visit(node.operand))
                    raise ValueError("Only numeric arithmetic is allowed.")

                try:
                    result = visit(tree)
                except ZeroDivisionError:
                    raise
                except OverflowError as exc:
                    raise ValueError("Result is too large.") from exc
                if isinstance(result, complex) or not isinstance(result, (int, float)):
                    raise ValueError("Result must be a real number.")
                if isinstance(result, float) and not __import__("math").isfinite(result):
                    raise ValueError("Result is not finite.")
                return result
        '''),
        "calculator_gui.py": textwrap.dedent('''\\
            """Modern keypad-style Tkinter calculator. Run: python calculator_gui.py"""
            import tkinter as tk
            from tkinter import messagebox
            from calculator import evaluate

            BG = "#101827"
            PANEL = "#182338"
            DISPLAY = "#0b1220"
            TEXT = "#f3f6fc"
            MUTED = "#9aa8bf"
            ACCENT = "#5b8cff"
            OPERATOR = "#263b5d"
            DANGER = "#7f3345"

            class CalculatorApp:
                def __init__(self, root):
                    self.root = root
                    self.expression = ""
                    self.just_evaluated = False
                    root.title("Python Calculator")
                    root.geometry("350x500")
                    root.minsize(320, 460)
                    root.configure(bg=BG)

                    tk.Label(root, text="PYTHON CALCULATOR", bg=BG, fg=MUTED,
                             font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(17, 8))
                    display = tk.Frame(root, bg=DISPLAY, highlightthickness=1, highlightbackground="#2a3850")
                    display.pack(fill="x", padx=14, pady=(0, 14))
                    self.preview = tk.StringVar(value="")
                    tk.Label(display, textvariable=self.preview, bg=DISPLAY, fg=MUTED,
                             anchor="e", font=("Segoe UI", 11)).pack(fill="x", padx=12, pady=(10, 0))
                    self.screen = tk.StringVar(value="0")
                    tk.Label(display, textvariable=self.screen, bg=DISPLAY, fg=TEXT,
                             anchor="e", font=("Segoe UI", 28, "bold")).pack(fill="x", padx=12, pady=(0, 10))

                    grid = tk.Frame(root, bg=BG)
                    grid.pack(fill="both", expand=True, padx=14, pady=(0, 14))
                    for col in range(4):
                        grid.columnconfigure(col, weight=1, uniform="col")
                    for row in range(5):
                        grid.rowconfigure(row, weight=1, uniform="row")
                    keys = [
                        [("AC", "clear", DANGER), ("⌫", "backspace", OPERATOR), ("%", "percent", OPERATOR), ("÷", "/", OPERATOR)],
                        [("7", "7", PANEL), ("8", "8", PANEL), ("9", "9", PANEL), ("×", "*", OPERATOR)],
                        [("4", "4", PANEL), ("5", "5", PANEL), ("6", "6", PANEL), ("−", "-", OPERATOR)],
                        [("1", "1", PANEL), ("2", "2", PANEL), ("3", "3", PANEL), ("+", "+", OPERATOR)],
                        [("+/−", "sign", PANEL), ("0", "0", PANEL), (".", ".", PANEL), ("=", "equals", ACCENT)],
                    ]
                    for r, row in enumerate(keys):
                        for c, (label, action, color) in enumerate(row):
                            btn = tk.Button(
                                grid, text=label, command=lambda a=action: self.press(a),
                                bg=color, fg=TEXT, activebackground=ACCENT, activeforeground=TEXT,
                                relief="flat", bd=0, cursor="hand2",
                                font=("Segoe UI", 14, "bold" if label in {"=", "AC"} else "normal"),
                                highlightthickness=0
                            )
                            btn.grid(row=r, column=c, sticky="nsew", padx=4, pady=4, ipadx=2, ipady=4)
                    root.bind("<Key>", self.on_key)
                    root.bind("<Return>", lambda _e: self.press("equals"))
                    root.bind("<KP_Enter>", lambda _e: self.press("equals"))
                    root.bind("<Escape>", lambda _e: self.press("clear"))
                    root.bind("<BackSpace>", lambda _e: self.press("backspace"))

                def refresh(self):
                    self.screen.set(self.expression or "0")

                def press(self, action):
                    if action == "clear":
                        self.expression = ""
                        self.preview.set("")
                        self.just_evaluated = False
                    elif action == "backspace":
                        if self.just_evaluated:
                            self.expression = ""
                            self.just_evaluated = False
                        else:
                            self.expression = self.expression[:-1]
                    elif action == "equals":
                        self.calculate()
                        return
                    elif action == "percent":
                        self.expression += "/100"
                        self.just_evaluated = False
                    elif action == "sign":
                        if self.expression.startswith("-"):
                            self.expression = self.expression[1:]
                        elif self.expression:
                            self.expression = "-" + self.expression
                        self.just_evaluated = False
                    else:
                        if self.just_evaluated and (action.isdigit() or action == "."):
                            self.expression = ""
                        self.just_evaluated = False
                        self.expression += "*" if action == "×" else action
                    self.refresh()

                def calculate(self):
                    if not self.expression:
                        return
                    shown = self.expression.replace("*", "×").replace("/", "÷").replace("-", "−")
                    try:
                        result = evaluate(self.expression)
                        if isinstance(result, float) and result.is_integer():
                            result = int(result)
                        self.preview.set(shown + " =")
                        self.expression = str(result)
                        self.screen.set(str(result))
                        self.just_evaluated = True
                    except (ValueError, ZeroDivisionError) as exc:
                        messagebox.showerror("Calculation error", str(exc))
                    except Exception:
                        messagebox.showerror("Calculation error", "This expression could not be calculated.")

                def on_key(self, event):
                    if event.char in "0123456789.+-*/%":
                        self.press(event.char)
                    elif event.keysym in ("Return", "KP_Enter"):
                        self.press("equals")
                    elif event.keysym == "BackSpace":
                        self.press("backspace")
                    elif event.keysym == "Escape":
                        self.press("clear")

            def main():
                root = tk.Tk()
                CalculatorApp(root)
                root.mainloop()

            if __name__ == "__main__":
                main()
        '''),
        "test_calculator.py": textwrap.dedent('''\\
            import unittest
            from calculator import evaluate

            class CalculatorTests(unittest.TestCase):
                def test_basic_operations_and_precedence(self):
                    self.assertEqual(evaluate("2+3*4"), 14)
                    self.assertEqual(evaluate("(2+3)*4"), 20)
                    self.assertEqual(evaluate("9-4"), 5)
                    self.assertEqual(evaluate("8/2"), 4)

                def test_decimal_unary_modulo_and_power(self):
                    self.assertEqual(evaluate("1.5+2.25"), 3.75)
                    self.assertEqual(evaluate("-5+2"), -3)
                    self.assertEqual(evaluate("10%3"), 1)
                    self.assertEqual(evaluate("2**3"), 8)

                def test_division_by_zero(self):
                    with self.assertRaises(ZeroDivisionError):
                        evaluate("5/0")

                def test_rejects_non_arithmetic_code(self):
                    for expression in ("__import__('os')", "2+open('x')", "x+1"):
                        with self.subTest(expression=expression):
                            with self.assertRaises(ValueError):
                                evaluate(expression)

                def test_rejects_oversized_exponent(self):
                    with self.assertRaises(ValueError):
                        evaluate("2**101")

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
        "from calculator import evaluate",
        "try:\n    from .calculator import evaluate\nexcept ImportError:\n    from calculator import evaluate",
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
        f"{folder}/__init__.py": '"""Calculator application package."""\nfrom .calculator import evaluate\n',
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
