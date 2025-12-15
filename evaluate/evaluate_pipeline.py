#!/usr/bin/env python3
"""
MULTI-COLLECTION EVALUATION (Docker)

- Reads persisted ChromaDB mounted at /chromadb (read-only)
- Auto-discovers collections and evaluates each one
- Builds unique questions from Chroma metadata (no CSV required)
- Retrieves top-k relevant context chunks per question from Chroma
  - NarrativeQA: retrieves within the SAME story (story_id)
  - HotpotQA: retrieves within the SAME question (question_id)
- Calls QA models through gRPC router
- Saves results per collection under /data/results/<collection_name>/

Env vars:
  CHROMA_DB_PATH=/chromadb
  ROUTER_ADDR=qa-router:50050
  MODELS="llama3,qwen2_5,mistral"
  TOP_K=6
  RESTRICT_TO_SAME_QUESTION=1
  MAX_SAMPLES=10 (or "None")
  EMBED_MODEL=all-MiniLM-L6-v2
  EVAL_COLLECTIONS="hotpotqa,narrativeqa"   # optional
  RAG_DEBUG=1  # optional
  GRPC_TIMEOUT_S=180
"""

import os
import time
import re
import grpc
import numpy as np
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import qa_pb2
import qa_pb2_grpc


# -----------------------------
# Configuration
# -----------------------------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/chromadb")

ROUTER_ADDR = os.getenv("ROUTER_ADDR", "qa-router:50050")

# UPDATED DEFAULT MODELS (router variant keys)
MODELS = [
    m.strip()
    for m in os.getenv("MODELS", "llama3,qwen2_5,mistral").split(",")
    if m.strip()
]

RESULTS_PATH = os.getenv("RESULTS_PATH", "/data/results")

TOP_K = int(os.getenv("TOP_K", "6"))
RESTRICT_TO_SAME_QUESTION = os.getenv("RESTRICT_TO_SAME_QUESTION", "1") == "1"

MAX_SAMPLES = os.getenv("MAX_SAMPLES", "100")  # "None" or integer
MAX_SAMPLES = None if str(MAX_SAMPLES).lower() == "none" else int(MAX_SAMPLES)

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

EVAL_COLLECTIONS = os.getenv("EVAL_COLLECTIONS", "").strip()
EVAL_COLLECTIONS = (
    [c.strip() for c in EVAL_COLLECTIONS.split(",") if c.strip()]
    if EVAL_COLLECTIONS
    else None
)

RAG_DEBUG = os.getenv("RAG_DEBUG", "0") == "1"

# NEW: longer default timeout for Ollama
GRPC_TIMEOUT_S = int(os.getenv("GRPC_TIMEOUT_S", "180"))


# -----------------------------
# Similarity helpers
# -----------------------------
def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# -----------------------------
# Connections
# -----------------------------
def connect_to_chromadb_client():
    print(f"Connecting to ChromaDB at {CHROMA_DB_PATH} ...")
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    print("Connected to ChromaDB")
    return client

def list_collections(client) -> list[str]:
    cols = client.list_collections()
    return [c.name for c in cols]

def get_collection(client, name: str):
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=name, embedding_function=embedding_fn)

def connect_to_router():
    print(f"Connecting to router at {ROUTER_ADDR} ...")
    channel = grpc.insecure_channel(ROUTER_ADDR)
    stub = qa_pb2_grpc.QAServerStub(channel)
    print("Connected to router")
    return stub


# -----------------------------
# Build question list from Chroma metadata
# -----------------------------
def load_questions_from_chroma(collection, limit=None) -> pd.DataFrame:
    """
    Build unique question list from chunk metadatas.

    Required metadata:
      - question_id
      - question
      - answer

    NarrativeQA fix:
      - If story_id exists in metadata, we keep it so retrieval can filter by story_id.
    """
    print("Building question list from Chroma metadata...")

    got = collection.get(include=["metadatas"])
    metas = got["metadatas"]

    seen = set()
    rows = []
    for m in metas:
        qid = str(m.get("question_id", "")).strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)

        rows.append(
            {
                "id": qid,
                "story_id": str(m.get("story_id", "")).strip(),
                "question": str(m.get("question", "")).strip(),
                "answer": str(m.get("answer", "")).strip(),
                "type": str(m.get("type", "")),
                "level": str(m.get("level", "")),
                "dataset": str(m.get("dataset", "")).strip().lower(),  # normalized
            }
        )
        if limit and len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} unique questions from Chroma")
    return df


# -----------------------------
# Retrieval (dataset-aware)
# -----------------------------
def build_context_from_chroma(
    collection,
    question: str,
    question_id: str | None,
    story_id: str | None,
    k: int
) -> str:
    """
    Dataset-aware retrieval filter:
      - NarrativeQA: filter by story_id (retrieve within story)
      - HotpotQA: filter by question_id (retrieve within question)
      - Else: unfiltered
    """
    where = None
    if RESTRICT_TO_SAME_QUESTION:
        sid = (story_id or "").strip()
        qid = (question_id or "").strip()
        if sid:
            where = {"story_id": sid}
        elif qid:
            where = {"question_id": qid}

    results = collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    parts = []
    for d, m in zip(docs, metas):
        ci = m.get("chunk_index", "")
        sid = m.get("story_id", "")
        qid = m.get("question_id", "")
        parts.append(f"[chunk {ci}] story_id={sid} question_id={qid}\n{d}")

    ctx = "\n\n---\n\n".join(parts).strip()

    if RAG_DEBUG:
        print("\n--- RAG DEBUG ---")
        print("where:", where)
        print("context_chars:", len(ctx))
        print("context_preview:\n", ctx[:500])
        print("--- END RAG DEBUG ---\n")

    return ctx


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_collection(collection_name: str, collection, stub) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"Evaluating collection: {collection_name}")
    print("=" * 70)
    print(f"Collection '{collection_name}' has {collection.count()} chunks")

    df_questions = load_questions_from_chroma(collection, limit=MAX_SAMPLES or None)

    if df_questions.empty:
        print(f"No questions found in collection '{collection_name}' (missing metadata?)")
        return pd.DataFrame()

    total_q = len(df_questions)
    print(
        f"Evaluating {total_q} questions x {len(MODELS)} models "
        f"(k={TOP_K}, restrict={RESTRICT_TO_SAME_QUESTION}, timeout={GRPC_TIMEOUT_S}s)"
    )

    answer_embedder = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    results = []
    pbar = tqdm(total=total_q * len(MODELS), desc=f"Evaluating {collection_name}")

    for _, row in df_questions.iterrows():
        qid = str(row.get("id", "")).strip()
        story_id = str(row.get("story_id", "")).strip()
        question = str(row.get("question", "")).strip()
        true_answer = str(row.get("answer", "")).strip()
        q_type = str(row.get("type", ""))
        q_level = str(row.get("level", ""))
        dataset = str(row.get("dataset", "")).strip().lower()

        context = build_context_from_chroma(
            collection,
            question=question,
            question_id=qid,
            story_id=story_id,
            k=TOP_K,
        )

        true_vec = (
            np.array(answer_embedder([true_answer])[0], dtype=np.float32)
            if true_answer
            else np.zeros(384, dtype=np.float32)
        )

        for model_variant in MODELS:
            try:
                req = qa_pb2.Question(question=question, context=context)

                start = time.time()
                resp = stub.Answer(
                    req,
                    metadata=[
                        ("variant", model_variant),      # router variant key
                        ("dataset", dataset),            # helpful now + later
                    ],
                    timeout=GRPC_TIMEOUT_S,
                )
                elapsed_ms = (time.time() - start) * 1000.0

                pred_answer = resp.answer or ""
                pred_vec = (
                    np.array(answer_embedder([pred_answer])[0], dtype=np.float32)
                    if pred_answer
                    else np.zeros_like(true_vec)
                )

                ans_cos = cosine_sim(true_vec, pred_vec)
                exact = 1 if _normalize_text(pred_answer) == _normalize_text(true_answer) and true_answer else 0

                results.append(
                    {
                        "collection": collection_name,
                        "dataset": dataset,
                        "story_id": story_id,
                        "question_id": qid,
                        "question": question,
                        "true_answer": true_answer,
                        "predicted_answer": pred_answer,
                        "model": model_variant,
                        "confidence": float(resp.confidence),
                        "answer_cosine_sim": float(ans_cos),
                        "exact_match": int(exact),
                        "type": q_type,
                        "level": q_level,
                        "retrieval_k": TOP_K,
                        "router_retrieval_ms": float(getattr(resp, "retrieval_ms", 0.0)),
                        "inference_ms": float(getattr(resp, "inference_ms", 0.0)),
                        "end_to_end_ms": float(getattr(resp, "end_to_end_ms", 0.0)),
                        "client_elapsed_ms": float(elapsed_ms),
                    }
                )

            except grpc.RpcError as e:
                results.append(
                    {
                        "collection": collection_name,
                        "dataset": dataset,
                        "story_id": story_id,
                        "question_id": qid,
                        "question": question,
                        "true_answer": true_answer,
                        "predicted_answer": "ERROR",
                        "model": model_variant,
                        "confidence": 0.0,
                        "answer_cosine_sim": 0.0,
                        "exact_match": 0,
                        "type": q_type,
                        "level": q_level,
                        "retrieval_k": TOP_K,
                        "router_retrieval_ms": 0.0,
                        "inference_ms": 0.0,
                        "end_to_end_ms": 0.0,
                        "client_elapsed_ms": 0.0,
                        "grpc_code": str(e.code()),
                        "grpc_details": str(e.details()),
                    }
                )

            pbar.update(1)

    pbar.close()
    df_results = pd.DataFrame(results)
    print(f"Evaluation complete for '{collection_name}'. Rows: {len(df_results)}")
    return df_results


# -----------------------------
# Saving
# -----------------------------
def save_results_for_collection(collection_name: str, df_results: pd.DataFrame):
    out_dir = os.path.join(RESULTS_PATH, collection_name)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")

    summary_path = os.path.join(out_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Evaluation Summary - {collection_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Models: {', '.join(MODELS)}\n")
        f.write(f"TOP_K: {TOP_K}\n")
        f.write(f"RESTRICT_TO_SAME_QUESTION: {RESTRICT_TO_SAME_QUESTION}\n")
        f.write(f"EMBED_MODEL: {EMBED_MODEL}\n")
        f.write(f"GRPC_TIMEOUT_S: {GRPC_TIMEOUT_S}\n")
        f.write(f"Rows: {len(df_results)}\n\n")

        for model in MODELS:
            md = df_results[df_results["model"] == model]
            f.write(f"\n{model}:\n")
            f.write(f"  Total: {len(md)}\n")
            if len(md) > 0:
                f.write(f"  Avg confidence: {md['confidence'].mean():.3f}\n")
                f.write(f"  Avg answer_cosine_sim: {md['answer_cosine_sim'].mean():.3f}\n")
                f.write(f"  Exact match rate: {md['exact_match'].mean():.3f}\n")
                f.write(f"  Avg inference_ms: {md['inference_ms'].mean():.1f}\n")
                f.write(f"  Avg end_to_end_ms: {md['end_to_end_ms'].mean():.1f}\n")

    print(f"Saved summary: {summary_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n" + "=" * 70)
    print("Multi-Collection Evaluation Pipeline (Chroma Retrieval)")
    print("=" * 70)

    client = connect_to_chromadb_client()
    stub = connect_to_router()

    all_cols = list_collections(client)
    print(f"Found collections: {all_cols}")

    if EVAL_COLLECTIONS is not None:
        cols = [c for c in all_cols if c in EVAL_COLLECTIONS]
        print(f"Restricting to EVAL_COLLECTIONS={EVAL_COLLECTIONS} -> {cols}")
    else:
        cols = all_cols

    if not cols:
        print("No collections to evaluate.")
        return

    for name in cols:
        try:
            col = get_collection(client, name)
        except Exception as e:
            print(f"Skipping collection '{name}' (failed to open): {e}")
            continue

        df_results = evaluate_collection(name, col, stub)
        if df_results is None or df_results.empty:
            print(f"No results for '{name}'.")
            continue

        save_results_for_collection(name, df_results)

    print("\n" + "=" * 70)
    print("All collection evaluations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
