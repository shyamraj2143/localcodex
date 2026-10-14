from agent.request import classify_request, references_existing_context


def test_short_hinglish_gui_report_is_a_bug_report_not_general_chat():
    message = "is game me main.py run karne par gui open nahi hua"
    assert classify_request(message, has_history=True, has_project=True) == "bug_report"
    assert references_existing_context(message)


def test_followup_and_explicit_read_only_explanation_are_distinguished():
    assert classify_request("ab isme sound add karo", has_history=True) == "follow_up"
    assert classify_request("sirf concept samjhao, code change mat karna") == "explanation"
    assert classify_request("explain what this function does") == "explanation"


def test_unrelated_general_question_is_not_treated_as_a_project_task():
    assert classify_request("What is the capital of Japan?", has_project=True) == "general"
