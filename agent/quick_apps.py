"""Deterministic Tkinter demo apps with small, testable logic modules."""
import re
import textwrap


def detect_quick_app(task):
    text = str(task or "").casefold()
    if re.search(r"\bcalculator\b", text) and re.search(r"\b(create|make|build|generate|banao|bana)\b", text):
        return "calculator"
    if re.search(r"\b(number guessing|guessing game|number game)\b", text) and re.search(r"\b(create|make|build|generate|banao|bana)\b", text):
        return "number_guessing_game"
    if re.search(r"\b(to-?do|task list|to do list)\b", text) and re.search(r"\b(create|make|build|generate|banao|bana)\b", text):
        return "todo_list_app"
    if re.search(r"\b(expense tracker|expense tracking|track expenses)\b", text) and re.search(r"\b(create|make|build|generate|banao|bana)\b", text):
        return "expense_tracker"
    return None


def quick_app_files(app):
    """Return folder-based GUI app files and unit tests; no third-party dependencies."""
    if app == "number_guessing_game":
        folder, title = app, "Number Guessing Game"
        logic = '''import random

def new_game(secret=None):
    return {"secret": random.randint(1, 100) if secret is None else secret, "attempts": 0, "finished": False}

def guess(state, value):
    if state["finished"]: return "finished"
    if not 1 <= value <= 100: return "out_of_range"
    state["attempts"] += 1
    if value == state["secret"]:
        state["finished"] = True
        return "correct"
    return "higher" if value < state["secret"] else "lower"
'''
        main = '''import tkinter as tk
from tkinter import messagebox
from app_logic import new_game, guess

class GuessingGame:
    def __init__(self, root):
        self.root = root
        root.title("Number Guessing Game")
        root.geometry("390x300")
        root.resizable(False, False)
        self.state = new_game()
        tk.Label(root, text="Guess the Number", font=("Segoe UI", 18, "bold")).pack(pady=(20, 8))
        tk.Label(root, text="I'm thinking of a number from 1 to 100.").pack()
        self.entry = tk.Entry(root, justify="center", font=("Segoe UI", 14), width=12)
        self.entry.pack(pady=12)
        self.entry.bind("<Return>", lambda _e: self.submit())
        tk.Button(root, text="Guess", command=self.submit, width=14).pack()
        self.status = tk.StringVar(value="Enter a number to start.")
        tk.Label(root, textvariable=self.status).pack(pady=10)
        self.count = tk.StringVar(value="Attempts: 0")
        tk.Label(root, textvariable=self.count).pack()
        tk.Button(root, text="New Game", command=self.restart).pack(pady=8)
    def submit(self):
        try: value = int(self.entry.get().strip())
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a whole number from 1 to 100.")
            return
        result = guess(self.state, value)
        self.status.set({"higher":"Too low — try higher!", "lower":"Too high — try lower!",
            "correct":"Correct! You won!", "out_of_range":"Choose a number from 1 to 100.",
            "finished":"Start a new game to play again."}[result])
        self.count.set(f"Attempts: {self.state['attempts']}")
        self.entry.delete(0, tk.END)
        if result == "correct":
            messagebox.showinfo("You won!", f"You guessed it in {self.state['attempts']} attempts.")
    def restart(self):
        self.state = new_game()
        self.status.set("New number selected. Good luck!")
        self.count.set("Attempts: 0")
        self.entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    GuessingGame(root)
    root.mainloop()
'''
        tests = '''import unittest
from app_logic import new_game, guess

class GuessTests(unittest.TestCase):
    def test_hints_and_attempt_count(self):
        s = new_game(secret=50)
        self.assertEqual(guess(s, 20), "higher")
        self.assertEqual(guess(s, 80), "lower")
        self.assertEqual(s["attempts"], 2)
    def test_win_and_finished_state(self):
        s = new_game(secret=42)
        self.assertEqual(guess(s, 42), "correct")
        self.assertEqual(guess(s, 42), "finished")
    def test_range_does_not_count(self):
        s = new_game(secret=42)
        self.assertEqual(guess(s, 101), "out_of_range")
        self.assertEqual(s["attempts"], 0)

if __name__ == "__main__": unittest.main()
'''
    elif app == "todo_list_app":
        folder, title = app, "To-Do List App"
        logic = '''def add_task(tasks, title):
    title = str(title).strip()
    if not title: raise ValueError("Task cannot be empty")
    tasks.append({"title": title, "done": False})
    return tasks

def toggle_task(tasks, index):
    if not 0 <= index < len(tasks): raise IndexError("Task index out of range")
    tasks[index]["done"] = not tasks[index]["done"]
    return tasks

def delete_task(tasks, index):
    if not 0 <= index < len(tasks): raise IndexError("Task index out of range")
    return tasks.pop(index)
'''
        main = '''import json
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from app_logic import add_task, toggle_task, delete_task
DATA = Path(__file__).with_name("tasks.json")

class TodoApp:
    def __init__(self, root):
        self.root = root
        root.title("To-Do List")
        root.geometry("480x470")
        self.tasks = self.load()
        tk.Label(root, text="My To-Do List", font=("Segoe UI", 18, "bold")).pack(pady=12)
        row = tk.Frame(root); row.pack(fill="x", padx=16)
        self.entry = tk.Entry(row, font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.entry.bind("<Return>", lambda _e: self.add())
        tk.Button(row, text="Add Task", command=self.add).pack(side="right")
        self.listbox = tk.Listbox(root, font=("Segoe UI", 11), height=15)
        self.listbox.pack(fill="both", expand=True, padx=16, pady=12)
        buttons = tk.Frame(root); buttons.pack(pady=(0, 12))
        tk.Button(buttons, text="Mark Done / Undo", command=self.toggle).pack(side="left", padx=4)
        tk.Button(buttons, text="Delete Selected", command=self.delete).pack(side="left", padx=4)
        self.render()
    def load(self):
        try:
            data = json.loads(DATA.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError): return []
    def save(self): DATA.write_text(json.dumps(self.tasks, indent=2), encoding="utf-8")
    def render(self):
        self.listbox.delete(0, tk.END)
        for t in self.tasks: self.listbox.insert(tk.END, ("✓ " if t["done"] else "○ ") + t["title"])
    def selected(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Select a task", "Choose a task first."); return None
        return selection[0]
    def add(self):
        try: add_task(self.tasks, self.entry.get())
        except ValueError as e: messagebox.showerror("Invalid task", str(e)); return
        self.entry.delete(0, tk.END); self.save(); self.render()
    def toggle(self):
        i = self.selected()
        if i is not None: toggle_task(self.tasks, i); self.save(); self.render(); self.listbox.selection_set(i)
    def delete(self):
        i = self.selected()
        if i is not None: delete_task(self.tasks, i); self.save(); self.render()

if __name__ == "__main__":
    root = tk.Tk(); TodoApp(root); root.mainloop()
'''
        tests = '''import unittest
from app_logic import add_task, toggle_task, delete_task

class TodoTests(unittest.TestCase):
    def test_add_validation(self):
        tasks = []
        with self.assertRaises(ValueError): add_task(tasks, " ")
        add_task(tasks, " Study Python ")
        self.assertEqual(tasks[0], {"title": "Study Python", "done": False})
    def test_toggle_and_delete(self):
        tasks = []; add_task(tasks, "Read"); toggle_task(tasks, 0)
        self.assertTrue(tasks[0]["done"])
        self.assertEqual(delete_task(tasks, 0)["title"], "Read")
        self.assertEqual(tasks, [])

if __name__ == "__main__": unittest.main()
'''
    elif app == "expense_tracker":
        folder, title = app, "Expense Tracker"
        logic = '''def add_expense(expenses, title, amount, category):
    title = str(title).strip()
    if not title: raise ValueError("Description cannot be empty")
    amount = float(amount)
    if amount <= 0: raise ValueError("Amount must be greater than zero")
    expenses.append({"title": title, "amount": round(amount, 2), "category": str(category).strip() or "Other"})
    return expenses

def total_expenses(expenses):
    return round(sum(float(item["amount"]) for item in expenses), 2)

def delete_expense(expenses, index):
    if not 0 <= index < len(expenses): raise IndexError("Expense index out of range")
    return expenses.pop(index)
'''
        main = '''import json
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from app_logic import add_expense, total_expenses, delete_expense
DATA = Path(__file__).with_name("expenses.json")

class ExpenseTracker:
    def __init__(self, root):
        self.root = root
        root.title("Expense Tracker"); root.geometry("600x500")
        self.expenses = self.load()
        tk.Label(root, text="Expense Tracker", font=("Segoe UI", 18, "bold")).pack(pady=12)
        form = tk.Frame(root); form.pack(fill="x", padx=16)
        self.description = tk.Entry(form, width=22); self.description.grid(row=0, column=0, padx=4, pady=5)
        self.amount = tk.Entry(form, width=12); self.amount.grid(row=0, column=1, padx=4, pady=5)
        self.category = ttk.Combobox(form, values=["Food","Transport","Study","Bills","Shopping","Other"], width=12)
        self.category.set("Other"); self.category.grid(row=0, column=2, padx=4, pady=5)
        tk.Button(form, text="Add Expense", command=self.add).grid(row=0, column=3, padx=4, pady=5)
        self.table = ttk.Treeview(root, columns=("description","amount","category"), show="headings", height=14)
        for key, label, width in [("description","Description",260),("amount","Amount",100),("category","Category",140)]:
            self.table.heading(key, text=label); self.table.column(key, width=width)
        self.table.pack(fill="both", expand=True, padx=16, pady=12)
        bottom = tk.Frame(root); bottom.pack(fill="x", padx=16, pady=(0,12))
        self.total = tk.StringVar()
        tk.Label(bottom, textvariable=self.total, font=("Segoe UI",13,"bold")).pack(side="left")
        tk.Button(bottom, text="Delete Selected", command=self.delete).pack(side="right")
        self.render()
    def load(self):
        try:
            data = json.loads(DATA.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError): return []
    def save(self): DATA.write_text(json.dumps(self.expenses, indent=2), encoding="utf-8")
    def render(self):
        for item in self.table.get_children(): self.table.delete(item)
        for i, e in enumerate(self.expenses):
            self.table.insert("", "end", iid=str(i), values=(e["title"], f'{e["amount"]:.2f}', e["category"]))
        self.total.set(f'Total expenses: {total_expenses(self.expenses):.2f}')
    def add(self):
        try: add_expense(self.expenses, self.description.get(), self.amount.get(), self.category.get())
        except (ValueError, TypeError) as e: messagebox.showerror("Invalid expense", str(e) or "Enter a positive amount."); return
        self.description.delete(0, tk.END); self.amount.delete(0, tk.END); self.save(); self.render()
    def delete(self):
        selected = self.table.selection()
        if not selected: messagebox.showinfo("Select an expense", "Choose an expense first."); return
        delete_expense(self.expenses, int(selected[0])); self.save(); self.render()

if __name__ == "__main__":
    root = tk.Tk(); ExpenseTracker(root); root.mainloop()
'''
        tests = '''import unittest
from app_logic import add_expense, total_expenses, delete_expense

class ExpenseTests(unittest.TestCase):
    def test_add_and_total(self):
        items = []; add_expense(items, "Lunch", "12.50", "Food"); add_expense(items, "Bus", 2.25, "Transport")
        self.assertEqual(total_expenses(items), 14.75)
    def test_invalid_values(self):
        with self.assertRaises(ValueError): add_expense([], "Lunch", 0, "Food")
        with self.assertRaises(ValueError): add_expense([], " ", 2, "Food")
    def test_delete(self):
        items = []; add_expense(items, "Book", 10, "Study")
        self.assertEqual(delete_expense(items, 0)["title"], "Book")
        self.assertEqual(total_expenses(items), 0)

if __name__ == "__main__": unittest.main()
'''
    else:
        raise ValueError("Unsupported quick app: " + str(app))

    readme = f"# {title}\n\nReady-to-run Python GUI application using Tkinter and the standard library.\n\nRun from this folder with: python main.py\n\nRun tests with: python -m unittest -v\n\nCore behavior is isolated in app_logic.py for automated testing. User data is stored locally as JSON where applicable.\n"
    launcher = '"""Launch the generated GUI app."""\nimport runpy\nfrom pathlib import Path\nrunpy.run_path(str(Path(__file__).resolve().parent / "' + folder + '" / "main.py"), run_name="__main__")\n'
    return {
        folder + "/main.py": main,
        folder + "/app_logic.py": logic,
        folder + "/test_app_logic.py": tests,
        folder + "/README.md": readme,
        "main.py": launcher,
    }
