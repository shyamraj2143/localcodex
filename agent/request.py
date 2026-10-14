"""Route ordinary questions separately from workspace debugging and edits."""

import re


BUG_PHRASES = (
    "not working", "doesn't work", "does not work", "failed", "failure", "error",
    "bug", "crash", "exception", "traceback", "not opening", "won't open",
    "didn't open", "did not open", "not open", "open nahi", "open nhi",
    "nahi hua", "nahi ho raha", "nahi chal", "kaam nahi", "fail ho raha",
    "issue aa", "problem aa", "problem hai", "run karne par", "run karne pe",
)
PROJECT_REFERENCES = (
    "this project", "that project", "this app", "this game", "this file", "this code",
    "the project", "the app", "the game", "the file", "isme", "is mein", "is game",
    "is app", "usme", "ab isme", "ye code", "ye file", "it still", "same error",
)
CODE_SIGNALS = re.compile(
    r"\b(code|python|javascript|function|class|project|repository|repo|file|folder|"
    r"app|application|game|gui|window|button|api|database|test|build|compile|"
    r"module|import|dependency|package|script|terminal|exception|traceback)\b|"
    r"(?:[\w./\\-]+\.(?:py|js|ts|json|toml|yml|yaml|html|css))",
    re.I,
)
EXPLAIN_ONLY = re.compile(
    r"\b(explain|describe|what does|how does|walk me through|samjhao|samjha do|"
    r"meaning of|concept)\b",
    re.I,
)
NO_CHANGE = re.compile(
    r"(?:\b(?:sirf|only)\b.*\b(?:explain|concept|no code)\b)|"
    r"\b(?:don't|do not|without)\s+(?:change|modify|edit|write)\b",
    re.I,
)


def classify_request(text, *, has_history=False, has_project=True):
    """Return general, explanation, bug_report, follow_up, or implementation."""
    message = str(text or "").strip()
    lowered = message.casefold()
    if not message:
        return "general"
    if NO_CHANGE.search(message):
        return "explanation"
    if any(phrase in lowered for phrase in BUG_PHRASES):
        return "bug_report"
    if EXPLAIN_ONLY.search(message):
        return "explanation"
    if any(phrase in lowered for phrase in PROJECT_REFERENCES) and (has_history or has_project):
        return "follow_up"
    if CODE_SIGNALS.search(message):
        return "implementation"
    if has_history and re.search(r"\b(it|this|that|same|again|ab|isme|usme)\b", lowered):
        return "follow_up"
    return "general"


def references_existing_context(text):
    lowered = str(text or "").casefold()
    return (
        any(phrase in lowered for phrase in PROJECT_REFERENCES)
        or bool(re.search(r"(?:[\w./\\-]+\.(?:py|js|ts|json|toml|yml|yaml|html|css))", lowered))
        or bool(re.search(r"\b(this|that|it|same|again|ab|isme|usme|ye|woh)\b", lowered))
    )
