#!/usr/bin/env python3
"""
Quick Mixed Evaluation (AUTO routing, fixed-per-collection sampling, no overwrite)

Goal:
- For each collection in EVAL_COLLECTIONS (or all collections), sample EXACTLY N questions
  (PER_COLLECTION_SAMPLES, default=300) from that collection’s metadata.
- Mix (shuffle) all sampled questions together.
- For each question, call router with variant=auto so router uses the trained MinIO artifact
  to pick a backend (llama3 / qwen2_5 / mistral).
- Save results to RESULTS_PATH/RUN_ID/evaluation_results.csv (does not overwrite your main eval).

Chroma compatibility:
- Some versions: collection.peek(limit=n) exists but does NOT accept include=
- Some versions: peek may not return metadatas
- Fallback: collection.get(limit=n, include=["metadatas"]) (bounded, avoids full scan)

Env (important):
- CHROMA_DB_PATH=/chromadb
- ROUTER_ADDR=qa-router:50050
- EVAL_COLLECTIONS=hotpotqa,narrativeqa,pubmedqa
- PER_COLLECTION_SAMPLES=300
- PEEK_LIMIT=5000  (increase if you can’t reach 300 uniques)
- TOP_K=6
- GRPC_TIMEOUT_S=60
- RESULTS_PATH=/data/results_quick
- RUN_ID=latest (or omit for timestamp)
- KEEP_COLS=... (optional)
"""

import os
import time
import re
import grpc
import json
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import qa_pb2
import qa_pb2_grpc


# -----------------------------
# Env
# -----------------------------
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/chromadb")
ROUTER_ADDR = os.getenv("ROUTER_ADDR", "qa-router:50050")

TOP_K = int(os.getenv("TOP_K", "6"))
GRPC_TIMEOUT_S = int(os.getenv("GRPC_TIMEOUT_S", "60"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RESTRICT_TO_SAME_QUESTION = os.getenv("RESTRICT_TO_SAME_QUESTION", "1") == "1"

RESULTS_PATH = os.getenv("RESULTS_PATH", "/data/results_quick")
RUN_ID = os.getenv("RUN_ID", "").strip() or time.strftime("%Y%m%d_%H%M%S")

JSON_LOGS = os.getenv("JSON_LOGS", "1") == "1"

KEEP_COLS = os.getenv("KEEP_COLS", "").strip()
KEEP_COLS = [c.strip() for c in KEEP_COLS.split(",") if c.strip()] if KEEP_COLS else None

EVAL_COLLECTIONS = os.getenv("EVAL_COLLECTIONS", "").strip()
EVAL_COLLECTIONS = ([c.strip() for c in EVAL_COLLECTIONS.split(",") if c.strip()] if EVAL_COLLECTIONS else None)

MIX_SEED = int(os.getenv("MIX_SEED", "42"))

# NEW: fixed per-collection sampling
PER_COLLECTION_SAMPLES = int(os.getenv("PER_COLLECTION_SAMPLES", "300"))
PEEK_LIMIT = int(os.getenv("PEEK_LIMIT", str(max(30000000, PER_COLLECTION_SAMPLES * 4))))


# -----------------------------
# Helpers
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def log_event(**kv):
    if JSON_LOGS:
        print(json.dumps(kv, ensure_ascii=False), flush=True)


def qhash(s: str) -> str:
    return hashlib.sha256((s or "").strip().lower().encode("utf-8")).hexdigest()[:12]


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
# Chroma / Router
# -----------------------------
def connect_to_chromadb_client():
    print(f"Connecting to ChromaDB at {CHROMA_DB_PATH} ...", flush=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    print("Connected to ChromaDB", flush=True)
    return client


def list_collections(client) -> list[str]:
    return [c.name for c in client.list_collections()]


def get_collection(client, name: str):
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(name=name, embedding_function=embedding_fn)


def connect_to_router():
    print(f"Connecting to router at {ROUTER_ADDR} ...", flush=True)
    channel = grpc.insecure_channel(ROUTER_ADDR)
    stub = qa_pb2_grpc.QAServerStub(channel)
    print("Connected to router", flush=True)
    return stub


# -----------------------------
# Bounded metadata fetch (Chroma version compatible)
# -----------------------------
def _peek_metadatas(collection, n: int):
    """
    Try to get metadatas via peek(limit=n). Some Chroma versions:
      - don't accept include=
      - may or may not return metadatas
    """
    try:
        got = collection.peek(limit=n)
    except Exception:
        return None

    if isinstance(got, dict) and got.get("metadatas"):
        return got["metadatas"]
    return None


def _get_metadatas(collection, n: int):
    """
    Fallback: collection.get with limit + include metadatas.
    (Avoid unbounded get())
    """
    got = collection.get(limit=n, include=["metadatas"])
    metas = got.get("metadatas") if isinstance(got, dict) else None
    return metas or []


def load_questions_from_chroma(collection, collection_name: str, limit: int) -> pd.DataFrame:
    """
    Return up to `limit` unique questions from the collection.
    We fetch up to PEEK_LIMIT metadatas to find enough unique question_ids.
    """
    n = max(PEEK_LIMIT, limit)

    metas = _peek_metadatas(collection, n)
    if metas is None:
        metas = _get_metadatas(collection, n)

    seen = set()
    rows = []
    for m in metas:
        qid = str(m.get("question_id", "")).strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)

        rows.append(
            {
                "collection": collection_name,
                "question_id": qid,
                "story_id": str(m.get("story_id", "")).strip(),
                "question": str(m.get("question", "")).strip(),
                "answer": str(m.get("answer", "")).strip(),
                "type": str(m.get("type", "")),
                "level": str(m.get("level", "")),
                "dataset": str(m.get("dataset", "")).strip().lower(),
            }
        )

        if len(rows) >= limit:
            break

    return pd.DataFrame(rows)


def build_context_from_chroma(collection, question: str, story_id: str, question_id: str, k: int):
    where = None
    if RESTRICT_TO_SAME_QUESTION:
        if story_id:
            where = {"story_id": story_id}
        elif question_id:
            where = {"question_id": question_id}

    t0 = time.time()
    results = collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
        include=["documents", "metadatas"],
    )
    retrieval_ms = (time.time() - t0) * 1000.0

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    parts = []
    for d, m in zip(docs, metas):
        ci = m.get("chunk_index", "")
        sid2 = m.get("story_id", "")
        qid2 = m.get("question_id", "")
        parts.append(f"[chunk {ci}] story_id={sid2} question_id={qid2}\n{d}")

    ctx = "\n\n---\n\n".join(parts).strip()
    return ctx, float(retrieval_ms)


# -----------------------------
# Fixed sampling: N per collection (e.g., 300/300/300)
# -----------------------------
def build_fixed_per_collection_list(client, cols: list[str]) -> pd.DataFrame:
    dfs = []
    print(f"\nSampling {PER_COLLECTION_SAMPLES} per collection (peek_limit={PEEK_LIMIT})", flush=True)

    for name in cols:
        try:
            col = get_collection(client, name)
            df = load_questions_from_chroma(col, name, limit=PER_COLLECTION_SAMPLES)
            if df.empty:
                print(f"Collection '{name}' returned 0 questions", flush=True)
                continue

            # Fill dataset if missing -> default to collection name
            df["dataset"] = df["dataset"].fillna("").astype(str).str.strip().str.lower()
            df.loc[df["dataset"] == "", "dataset"] = name.strip().lower()

            # If we didn't reach target, warn (raise peek_limit if this happens)
            if len(df) < PER_COLLECTION_SAMPLES:
                print(
                    f"WARNING: '{name}' only yielded {len(df)}/{PER_COLLECTION_SAMPLES} unique questions. "
                    f"Increase PEEK_LIMIT to scan more metadatas.",
                    flush=True,
                )

            print(f"Loaded {len(df)} questions from {name}", flush=True)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {name}: {e}", flush=True)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    # Shuffle across collections so routing switches
    out = out.sample(frac=1.0, random_state=MIX_SEED).reset_index(drop=True)

    print("\nFinal counts by collection:", flush=True)
    print(out["collection"].value_counts(dropna=False).to_string(), flush=True)

    print("\nFinal counts by dataset:", flush=True)
    print(out["dataset"].value_counts(dropna=False).to_string(), flush=True)

    return out


# -----------------------------
# Evaluate (AUTO routing)
# -----------------------------
def evaluate_mixed_questions(client, stub, cols: list[str]) -> pd.DataFrame:
    qdf = build_fixed_per_collection_list(client, cols)
    if qdf.empty:
        print("No questions found.", flush=True)
        return pd.DataFrame()

    answer_embedder = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    rows = []
    pbar = tqdm(total=len(qdf), desc="quick fixed-per-collection")

    col_cache = {}

    for _, r in qdf.iterrows():
        collection_name = str(r["collection"])
        if collection_name not in col_cache:
            col_cache[collection_name] = get_collection(client, collection_name)
        collection = col_cache[collection_name]

        qid = str(r["question_id"])
        sid = str(r.get("story_id", ""))
        q = str(r.get("question", ""))
        gold = str(r.get("answer", ""))
        q_type = str(r.get("type", ""))
        q_level = str(r.get("level", ""))

        dataset = str(r.get("dataset", "")).strip().lower()
        if not dataset:
            dataset = collection_name.strip().lower() or "unknown"

        qid_hash = qhash(q)

        ctx, retr_ms = build_context_from_chroma(collection, q, sid, qid, TOP_K)

        true_vec = (
            np.array(answer_embedder([gold])[0], dtype=np.float32)
            if gold else np.zeros(384, dtype=np.float32)
        )

        t0 = time.time()
        try:
            # IMPORTANT: do NOT force a backend. Let router decide from MinIO-trained artifact.
            call_md = [
                ("dataset", dataset),
                ("variant", "auto"),
                ("retrieval_k", str(TOP_K)),
            ]

            resp, call = stub.Answer.with_call(
                qa_pb2.Question(question=q, context=ctx),
                metadata=call_md,
                timeout=GRPC_TIMEOUT_S,
            )
            grpc_ms = (time.time() - t0) * 1000.0
            pred = resp.answer or ""

            trail = dict(call.trailing_metadata() or [])
            routed_variant = trail.get("routed-variant", "")
            routed_by = trail.get("routed-by", "")

            pred_vec = (
                np.array(answer_embedder([pred])[0], dtype=np.float32)
                if pred else np.zeros_like(true_vec)
            )
            cos = cosine_sim(true_vec, pred_vec)
            em = 1 if (gold and _normalize_text(pred) == _normalize_text(gold)) else 0

            row = {
                "ts_ms": now_ms(),
                "collection": collection_name,
                "dataset": dataset,
                "story_id": sid,
                "question_id": qid,
                "question_hash": qid_hash,

                "question": q,
                "true_answer": gold,
                "predicted_answer": pred,

                "requested_variant": "auto",
                "routed_variant": routed_variant,
                "routed_by": routed_by,

                "retrieval_k": TOP_K,
                "local_retrieval_ms": float(retr_ms),
                "grpc_elapsed_ms": float(grpc_ms),
                "router_retrieval_ms": float(getattr(resp, "retrieval_ms", 0.0)),
                "inference_ms": float(getattr(resp, "inference_ms", 0.0)),
                "end_to_end_ms": float(getattr(resp, "end_to_end_ms", 0.0)),

                "prompt_chars": len(q),
                "context_chars": len(ctx),
                "answer_chars": len(pred),

                "exact_match": int(em),
                "answer_cosine_sim": float(cos),

                "type": q_type,
                "level": q_level,
            }
            rows.append(row)

            log_event(
                event="qa_auto",
                collection=collection_name,
                dataset=dataset,
                question_id=qid,
                routed_variant=routed_variant,
                routed_by=routed_by,
                grpc_elapsed_ms=float(grpc_ms),
                inference_ms=float(getattr(resp, "inference_ms", 0.0)),
                exact_match=int(em),
                answer_cosine_sim=float(cos),
            )

        except grpc.RpcError as e:
            rows.append({
                "ts_ms": now_ms(),
                "collection": collection_name,
                "dataset": dataset,
                "story_id": sid,
                "question_id": qid,
                "question_hash": qid_hash,
                "question": q,
                "true_answer": gold,
                "predicted_answer": "ERROR",

                "requested_variant": "auto",
                "routed_variant": "",
                "routed_by": "",

                "retrieval_k": TOP_K,
                "local_retrieval_ms": float(retr_ms),
                "grpc_elapsed_ms": 0.0,
                "router_retrieval_ms": 0.0,
                "inference_ms": 0.0,
                "end_to_end_ms": 0.0,
                "prompt_chars": len(q),
                "context_chars": len(ctx),
                "answer_chars": 0,
                "exact_match": 0,
                "answer_cosine_sim": 0.0,
                "type": q_type,
                "level": q_level,
                "grpc_code": str(e.code()),
                "grpc_details": str(e.details()),
            })

        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)

    if KEEP_COLS:
        df = df[[c for c in KEEP_COLS if c in df.columns]]

    return df


def save_results(df: pd.DataFrame):
    out_dir = os.path.join(RESULTS_PATH, RUN_ID)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}", flush=True)

    if "routed_variant" in df.columns:
        print("\n=== Routed variant counts ===", flush=True)
        print(df["routed_variant"].value_counts(dropna=False).to_string(), flush=True)

    if "routed_by" in df.columns:
        print("\n=== Routed-by counts ===", flush=True)
        print(df["routed_by"].value_counts(dropna=False).to_string(), flush=True)

    if "dataset" in df.columns:
        print("\n=== Dataset distribution ===", flush=True)
        print(df["dataset"].value_counts(dropna=False).to_string(), flush=True)

    if "collection" in df.columns:
        print("\n=== Collection distribution ===", flush=True)
        print(df["collection"].value_counts(dropna=False).to_string(), flush=True)


def main():
    print("Quick Mixed Evaluation (AUTO routing)", flush=True)
    print("RESULTS_PATH:", RESULTS_PATH, flush=True)
    print("RUN_ID:", RUN_ID, flush=True)
    print("TOP_K:", TOP_K, flush=True)
    print("GRPC_TIMEOUT_S:", GRPC_TIMEOUT_S, flush=True)
    print("PER_COLLECTION_SAMPLES:", PER_COLLECTION_SAMPLES, flush=True)
    print("PEEK_LIMIT:", PEEK_LIMIT, flush=True)
    print("EVAL_COLLECTIONS:", EVAL_COLLECTIONS, flush=True)

    client = connect_to_chromadb_client()
    stub = connect_to_router()

    all_cols = list_collections(client)
    if EVAL_COLLECTIONS:
        cols = [c for c in all_cols if c in EVAL_COLLECTIONS]
    else:
        cols = all_cols

    print("Collections:", cols, flush=True)

    if not cols:
        print("No collections to evaluate.", flush=True)
        return

    df = evaluate_mixed_questions(client, stub, cols)
    if df.empty:
        print("No results.", flush=True)
        return

    save_results(df)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
