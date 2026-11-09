"""
api.py - FastAPI wrapper around the agent.
...
"""
# The text in triple quotes at the top is a "docstring": a note describing the file.
# Python ignores it when running.

from fastapi import FastAPI            # FastAPI: the library that turns Python functions into web endpoints
from pydantic import BaseModel         # BaseModel: lets us describe the shape of the data (and checks it)

from agent import run_agent            # our agent function from agent.py

app = FastAPI(title="Company Policy Agent")
# Creates the web application object. Uvicorn looks for exactly this name:
# "api:app" in the command means file api.py, object app.
# The title appears at the top of the /docs page.


class ChatRequest(BaseModel):          # describes what a caller must SEND to /chat
    session_id: str                    # text: which conversation this is (same ID = same memory)
    message: str                       # text: the user's question


class ChatResponse(BaseModel):         # describes what /chat sends BACK
    answer: str                        # the agent's answer text
    tools_used: list[str]              # a list of tool names, e.g. ["lookup_employee"]
    steps: int                         # a whole number: how many steps were used
    hit_step_cap: bool                 # True or False: did it hit the step limit


@app.get("/health")                    # "decorator": when someone sends GET to /health, run the function below
def health():
    """Used by Docker or Kubernetes to check the service is alive."""
    return {"status": "ok"}            # returned as JSON: {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)   # when someone sends POST to /chat, run chat()
                                                  # response_model: the reply must match ChatResponse
def chat(request: ChatRequest):        # FastAPI reads the incoming JSON, checks it against ChatRequest,
                                       # and passes it in as `request`
    return run_agent(request.session_id, request.message)
                                       # call the agent with the two fields; the dictionary it
                                       # returns is checked against ChatResponse and sent back