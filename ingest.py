"""
ingest.py - Step 1: read the policy documents, chunk them, embed them,
and store them in ChromaDB.

Run once (and again whenever the documents in data/ change):
    python ingest.py
"""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
CHROMA_PATH = str(BASE / "chroma_db")
COLLECTION = "policies"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_chunks(data_dir):
    """Split every .txt file into paragraph-level chunks.

    Returns three parallel lists: chunk texts, ids, and metadata (source file name).
    """
    texts, ids, metadatas = [], [], []
    for path in sorted(data_dir.glob("*.txt")):
        paragraphs = path.read_text(encoding="utf-8").split("\n\n")
        chunk_number = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:  # skip empty or title-only lines
                continue
            texts.append(paragraph)
            ids.append(f"{path.stem}-{chunk_number}")
            metadatas.append({"source": path.name})
            chunk_number += 1
    return texts, ids, metadatas


def main():
    texts, ids, metadatas = load_chunks(DATA_DIR)
    print(f"Loaded {len(texts)} chunks from {DATA_DIR}")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts).tolist()
    print(f"Created embeddings, each with {len(embeddings[0])} dimensions")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)  # start fresh so re-running is safe
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Stored {collection.count()} chunks in ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
