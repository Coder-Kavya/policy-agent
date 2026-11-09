# Company Policy Agent

An agentic upgrade of a basic RAG pipeline. The agent answers policy questions from
retrieved documents, looks up employee data, and escalates to a human by creating a
ticket when it cannot answer. Prototype on synthetic data, not a production system.

## Run locally

    pip install -r requirements.txt
    export OPENAI_API_KEY=your_key_here
    python ingest.py          # chunk, embed, store in ChromaDB
    python agent.py           # chat in the terminal
    python evaluate.py        # score 15 test cases
    uvicorn api:app --reload  # API at http://127.0.0.1:8000/docs

## Run in Docker

    docker build -t policy-agent .
    docker run -p 8000:8000 -e OPENAI_API_KEY=your_key policy-agent

## Design

- Retrieval: sentence-transformers (all-MiniLM-L6-v2) + ChromaDB, cosine distance
- Agent: explicit loop on OpenAI tool calling (gpt-4o-mini), 3 tools, 5-step cap
- Memory: per-session message history (in-process, short-term)
- Safety: answers only from retrieved text, escalates to a human instead of guessing
- Evaluation: 15 cases scoring tool choice and answer facts

## Known limits

In-memory sessions are lost on restart and do not scale across replicas. Employee data and
tickets are mock files. No authentication, PII redaction, or monitoring yet.
