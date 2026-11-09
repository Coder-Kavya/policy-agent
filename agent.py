"""
agent.py - the agent loop, written directly on OpenAI tool calling.

The loop:
  1. Send the conversation + tool descriptions to the model.
  2. If the model asks for tool calls -> run them, add results, go back to 1.
  3. If the model answers in plain text -> that is the final answer.
  4. Stop after MAX_STEPS so the agent can never loop forever.
"""
import json

from openai import OpenAI

from tools import TOOL_FUNCS, TOOL_SCHEMAS

MODEL = "gpt-4o-mini"
MAX_STEPS = 5

SYSTEM_PROMPT = """You are an internal company assistant for employees.
Rules:
1. Answer company-policy questions ONLY from the results of the search_policy tool. Never use your own knowledge for company facts.
2. For questions about a specific employee (leave balance, team, manager), call lookup_employee with the employee ID the user gave. If no ID was given, ask for it.
3. If search_policy returns any_relevant = false, or the question is outside company policy, do NOT guess. Call create_ticket to escalate to a human, then tell the user a ticket was created and give the ticket ID.
4. Mention the source document name for every policy fact you state.
5. Keep answers short and factual."""

_client = None

# Short-term memory: session_id -> the list of messages so far in that conversation.
SESSIONS = {}


def get_client():
    """Create the OpenAI client once. It reads the OPENAI_API_KEY environment variable."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def run_agent(session_id, user_message, max_steps=MAX_STEPS):
    """Run the agent for one user message and return the answer plus a trace."""
    messages = SESSIONS.setdefault(
        session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
    )
    messages.append({"role": "user", "content": user_message})
    tools_used = []

    for step in range(1, max_steps + 1):
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0,  # deterministic: we want facts, not creativity
        )
        message = response.choices[0].message
        print(f"\n[STEP {step}] tool requests: "
                f"{[c.function.name for c in message.tool_calls] if message.tool_calls else 'none, final answer'}")

        # Case A: no tool calls -> this is the final answer.
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return {
                "answer": message.content,
                "tools_used": tools_used,
                "steps": step,
                "hit_step_cap": False,
            }

        # Case B: the model wants tools. Record its request, then run each tool.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments)
                result = TOOL_FUNCS[name](**arguments)
            except Exception as error:  # tool failures go back to the model, not a crash
                result = {"error": str(error)}
            tools_used.append(name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

            print(f"   -> ran {name} with {call.function.arguments}")
            print(f"   -> returned: {str(result)[:150]}")

    # Step cap reached without a final answer: escalate to a human instead of looping on.
    ticket = TOOL_FUNCS["create_ticket"](
        summary=f"Agent step limit reached for: {user_message}", department="General"
    )
    tools_used.append("create_ticket")
    answer = (
        "I could not complete this request within my step limit, so I created "
        f"ticket {ticket['ticket_id']} for a human to follow up."
    )
    messages.append({"role": "assistant", "content": answer})
    return {
        "answer": answer,
        "tools_used": tools_used,
        "steps": max_steps,
        "hit_step_cap": True,
    }


def main():
    """Small command-line chat to try the agent by hand."""
    session_id = "demo"
    print("Company assistant. Type 'quit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "quit":
            break
        result = run_agent(session_id, question)
        print(f"Agent: {result['answer']}")
        print(f"  [tools: {result['tools_used']} | steps: {result['steps']}]")


if __name__ == "__main__":
    main()
