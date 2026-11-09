"""
tools.py - the three tools the agent can call.

1. search_policy    - semantic search over the policy documents (your RAG retriever)
2. lookup_employee  - fetch personal data (mock database: employees.json)
3. create_ticket    - escalate to a human (mock: writes to tickets.json)
"""
import json
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
CHROMA_PATH = str(BASE / "chroma_db")
COLLECTION = "policies"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMPLOYEES_FILE = BASE / "employees.json"
TICKETS_FILE = BASE / "tickets.json"

# Cosine distance = 1 - cosine similarity. A smaller distance means a closer match.
# Chunks further away than this are marked "not relevant". Tune it using evaluate.py.
RELEVANCE_MAX_DISTANCE = 0.65

_model = None
_collection = None


def _get_model():
    """Load the embedding model once, on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection():
    """Open the ChromaDB collection once, on first use."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION)
    return _collection


def search_policy(query, k=3):
    """Return the top-k policy chunks for a question, with distance and a relevant flag."""
    query_embedding = _get_model().encode([query]).tolist()
    found = _get_collection().query(query_embeddings=query_embedding, n_results=k)

    results = []
    for text, meta, distance in zip(
        found["documents"][0], found["metadatas"][0], found["distances"][0]
    ):
        results.append(
            {
                "source": meta["source"],
                "text": text,
                "distance": round(distance, 3),
                "relevant": distance <= RELEVANCE_MAX_DISTANCE,
            }
        )
    return {"results": results, "any_relevant": any(r["relevant"] for r in results)}


def lookup_employee(employee_id):
    """Return the employee record, or an error if the ID does not exist."""
    employees = json.loads(EMPLOYEES_FILE.read_text(encoding="utf-8"))
    record = employees.get(employee_id.strip().upper())
    if record is None:
        return {"error": f"Employee {employee_id} not found"}
    return {"employee_id": employee_id.strip().upper(), **record}


def create_ticket(summary, department="General"):
    """Create an escalation ticket for a human to handle."""
    tickets = []
    if TICKETS_FILE.exists():
        tickets = json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
    ticket_id = "TCK-" + uuid.uuid4().hex[:4].upper()
    tickets.append({"ticket_id": ticket_id, "summary": summary, "department": department})
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
    return {"ticket_id": ticket_id, "status": "created", "department": department}


# Maps the tool name the model returns to the Python function that runs it.
TOOL_FUNCS = {
    "search_policy": search_policy,
    "lookup_employee": lookup_employee,
    "create_ticket": create_ticket,
}

# Descriptions the model reads to decide which tool to call and with which arguments.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search the company policy documents (HR, engineering, onboarding, "
                "product, security). Use this for any question about company rules or "
                "information. Returns chunks with a 'relevant' flag and 'any_relevant'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_employee",
            "description": (
                "Look up one employee's personal data (name, team, manager, leave "
                "balance) by employee ID such as E101."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "e.g. E101"}
                },
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": (
                "Escalate to a human by creating a ticket. Use when the question is "
                "outside company policy or the search found nothing relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What the user needs"},
                    "department": {
                        "type": "string",
                        "description": "HR, Engineering, Security, Product or General",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]
