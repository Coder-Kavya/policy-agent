"""
evaluate.py - run 15 test questions through the agent and score them.

Two checks per question:
  tool_ok   - did the agent call the tools we expected?
  answer_ok - does the answer contain the expected facts? (skipped for out-of-scope cases)
A case passes only if both checks pass.

Run:  python evaluate.py     (needs OPENAI_API_KEY and a finished 'python ingest.py')
"""
from agent import run_agent

# category: policy = answer from documents | personal = needs employee lookup
#           mixed = needs both            | out_of_scope = must escalate with a ticket
CASES = [
    {"q": "How many days of annual leave do employees get?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["18"]},
    {"q": "How many approvals does a pull request need before merging?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["2"]},
    {"q": "What is the minimum password length?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["14"]},
    {"q": "Within how long must a security incident be reported?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["1 hour"]},
    {"q": "What is the first response time for Enterprise support?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["1 hour"]},
    {"q": "When does a new joiner get their laptop?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["day one"]},
    {"q": "How many days a week can I work from home?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["3"]},
    {"q": "What unit test coverage is required for new code?", "category": "policy",
     "tools": {"search_policy"}, "must_contain": ["80"]},
    {"q": "I am employee E101. How many leave days do I have left?", "category": "personal",
     "tools": {"lookup_employee"}, "must_contain": ["12"]},
    {"q": "I am employee E102. Who is my manager?", "category": "personal",
     "tools": {"lookup_employee"}, "must_contain": ["rohan mehta"]},
    {"q": "My employee ID is E103. Which team am I in?", "category": "personal",
     "tools": {"lookup_employee"}, "must_contain": ["security"]},
    {"q": "I am E102. How many leave days can I carry forward, and how many do I have now?",
     "category": "mixed", "tools": {"search_policy", "lookup_employee"},
     "must_contain": ["5", "7"]},
    {"q": "What is the company's stock price today?", "category": "out_of_scope",
     "tools": {"create_ticket"}, "must_contain": []},
    {"q": "Can you recommend a good restaurant near the office?", "category": "out_of_scope",
     "tools": {"create_ticket"}, "must_contain": []},
    {"q": "What is the CEO's salary?", "category": "out_of_scope",
     "tools": {"create_ticket"}, "must_contain": []},
]


def main():
    passed = 0
    tool_correct = 0
    total_steps = 0
    failures = []

    print(f"{'#':>2}  {'category':<13} {'tools':<5} {'answer':<6} {'steps':<5} question")
    for number, case in enumerate(CASES, start=1):
        result = run_agent(f"eval-{number}", case["q"])  # fresh session per case

        tool_ok = case["tools"].issubset(set(result["tools_used"]))
        answer_text = (result["answer"] or "").lower()
        answer_ok = all(fact in answer_text for fact in case["must_contain"])

        tool_correct += tool_ok
        total_steps += result["steps"]
        if tool_ok and answer_ok:
            passed += 1
        else:
            failures.append((number, case["q"], result))

        print(f"{number:>2}  {case['category']:<13} {'ok' if tool_ok else 'FAIL':<5} "
              f"{'ok' if answer_ok else 'FAIL':<6} {result['steps']:<5} {case['q']}")

    total = len(CASES)
    print("\n--- Summary ---")
    print(f"Passed (tool + answer): {passed}/{total} = {100 * passed / total:.0f}% of {total} cases")
    print(f"Correct tool choice:    {tool_correct}/{total} = {100 * tool_correct / total:.0f}% of {total} cases")
    print(f"Average steps per case: {total_steps / total:.1f}")

    if failures:
        print("\n--- Failures (read these, they show what to fix) ---")
        for number, question, result in failures:
            print(f"#{number}: {question}")
            print(f"   tools used: {result['tools_used']}")
            print(f"   answer:     {result['answer']}")

if __name__ == "__main__":
    main()
