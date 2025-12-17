#!/usr/bin/env python3
"""
MULTI-COLLECTION EVALUATION (Docker)

Fixes:
- dataset is forced non-empty (fallback to collection_name)
- training runs after eval, then uploads model to MinIO (optional)
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

TRAIN_ROUTER_MODEL = os.getenv("TRAIN_ROUTER_MODEL", "0") == "1"
ROUTER_MODEL_PATH = os.getenv("ROUTER_MODEL_PATH", "/data/router_model/router_model.joblib")
ROUTER_TRAIN_GLOB = os.getenv("ROUTER_TRAIN_GLOB", "/data/results/**/evaluation_results.csv")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/chromadb")
ROUTER_ADDR = os.getenv("ROUTER_ADDR", "qa-router:50050")

MODELS = [m.strip() for m in os.getenv("MODELS", "llama3,qwen2_5,mistral").split(",") if m.strip()]
RESULTS_PATH = os.getenv("RESULTS_PATH", "/data/results")
TOP_K = int(os.getenv("TOP_K", "6"))
RESTRICT_TO_SAME_QUESTION = os.getenv("RESTRICT_TO_SAME_QUESTION", "1") == "1"

MAX_SAMPLES = os.getenv("MAX_SAMPLES", "1000")
MAX_SAMPLES = None if str(MAX_SAMPLES).lower() == "none" else int(MAX_SAMPLES)

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

EVAL_COLLECTIONS = os.getenv("EVAL_COLLECTIONS", "pubmedqa").strip()
EVAL_COLLECTIONS = ([c.strip() for c in EVAL_COLLECTIONS.split(",") if c.strip()] if EVAL_COLLECTIONS else None)

RAG_DEBUG = os.getenv("RAG_DEBUG", "0") == "1"
GRPC_TIMEOUT_S = int(os.getenv("GRPC_TIMEOUT_S", "180"))

JSON_LOGS = os.getenv("JSON_LOGS", "1") == "1"
RETR_CACHE_TTL_S = int(os.getenv("RETR_CACHE_TTL_S", "3600"))
TRIM_CHUNKS = os.getenv("TRIM_CHUNKS", "0") == "1"
MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", "1000"))

_retr_cache = {}  # key -> (expires_at, ctx_string)


def now_ms() -> int:
    return int(time.time() * 1000)

def qhash(s: str) -> str:
    return hashlib.sha256((s or "").strip().lower().encode("utf-8")).hexdigest()[:12]

def log_event(**kv):
    if JSON_LOGS:
        print(json.dumps(kv, ensure_ascii=False), flush=True)

def norm_q(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def cache_key(question: str, collection: str, k: int, restrict: bool, story_id: str, question_id: str) -> str:
    base = f"{collection}|k={k}|restrict={int(restrict)}|sid={story_id}|qid={question_id}|{norm_q(question)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

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


def load_questions_from_chroma(collection, limit=None) -> pd.DataFrame:
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
                "dataset": str(m.get("dataset", "")).strip().lower(),
            }
        )
        if limit and len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    print(f"Built {len(df)} unique questions from Chroma")
    return df


def trim_chunk(text: str, max_chars: int = 1000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"

def build_context_from_chroma(
    collection,
    collection_name: str,
    question: str,
    question_id: str | None,
    story_id: str | None,
    k: int,
    question_hash: str,
) -> tuple[str, float, bool]:
    where = None
    sid = (story_id or "").strip()
    qid = (question_id or "").strip()

    if RESTRICT_TO_SAME_QUESTION:
        if sid:
            where = {"story_id": sid}
        elif qid:
            where = {"question_id": qid}

    ck = cache_key(question, collection_name, k, RESTRICT_TO_SAME_QUESTION, sid, qid)
    item = _retr_cache.get(ck)
    if item:
        exp, cached_ctx = item
        if time.time() <= exp:
            log_event(
                ts_ms=now_ms(),
                event="retrieval",
                collection=collection_name,
                question_id=qid,
                question_hash=question_hash,
                retrieval_k=k,
                retrieval_ms=0.0,
                retr_cache_hit=True,
                where=where,
                context_chars=len(cached_ctx),
            )
            return cached_ctx, 0.0, True
        _retr_cache.pop(ck, None)

    t0 = time.time()
    results = collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    retrieval_ms = (time.time() - t0) * 1000.0

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    parts = []
    for d, m in zip(docs, metas):
        ci = m.get("chunk_index", "")
        sid2 = m.get("story_id", "")
        qid2 = m.get("question_id", "")
        doc_text = d
        if TRIM_CHUNKS:
            doc_text = trim_chunk(doc_text, MAX_CHARS_PER_CHUNK)
        parts.append(f"[chunk {ci}] story_id={sid2} question_id={qid2}\n{doc_text}")

    ctx = "\n\n---\n\n".join(parts).strip()
    _retr_cache[ck] = (time.time() + RETR_CACHE_TTL_S, ctx)

    if RAG_DEBUG:
        print("\n--- RAG DEBUG ---")
        print("where:", where)
        print("retrieval_ms:", retrieval_ms)
        print("context_chars:", len(ctx))
        print("context_preview:\n", ctx[:500])
        print("--- END RAG DEBUG ---\n")

    log_event(
        ts_ms=now_ms(),
        event="retrieval",
        collection=collection_name,
        question_id=qid,
        question_hash=question_hash,
        retrieval_k=k,
        retrieval_ms=float(retrieval_ms),
        retr_cache_hit=False,
        where=where,
        context_chars=len(ctx),
        trim_chunks=bool(TRIM_CHUNKS),
        max_chars_per_chunk=int(MAX_CHARS_PER_CHUNK),
    )

    return ctx, float(retrieval_ms), False


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

        # ✅ force non-empty dataset (fixes training + routing features)
        dataset_raw = str(row.get("dataset", "")).strip().lower()
        dataset = dataset_raw or collection_name.lower()
        dataset = str(row.get("dataset", "")).strip().lower()
        if not dataset:
            dataset = collection_name.strip().lower() or "unknown"
        
                
        qid_hash = qhash(question)

        ctx, local_retrieval_ms, retr_cache_hit = build_context_from_chroma(
            collection,
            collection_name=collection_name,
            question=question,
            question_id=qid,
            story_id=story_id,
            k=TOP_K,
            question_hash=qid_hash,
        )

        true_vec = (
            np.array(answer_embedder([true_answer])[0], dtype=np.float32)
            if true_answer
            else np.zeros(384, dtype=np.float32)
        )

        for model_variant in MODELS:
            t_total0 = now_ms()
            pred_answer = ""

            try:
                req = qa_pb2.Question(question=question, context=ctx)

                t0 = time.time()
                resp = stub.Answer(
                    req,
                    metadata=[
                        ("variant", model_variant),
                        ("dataset", dataset),  # ✅ always non-empty
                    ],
                    timeout=GRPC_TIMEOUT_S,
                )
                grpc_elapsed_ms = (time.time() - t0) * 1000.0

                pred_answer = resp.answer or ""

                pred_vec = (
                    np.array(answer_embedder([pred_answer])[0], dtype=np.float32)
                    if pred_answer
                    else np.zeros_like(true_vec)
                )

                ans_cos = cosine_sim(true_vec, pred_vec)
                exact = 1 if _normalize_text(pred_answer) == _normalize_text(true_answer) and true_answer else 0

                row_out = {
                    "collection": collection_name,
                    "dataset": dataset,  # ✅ always present
                    "story_id": story_id,
                    "question_id": qid,
                    "question": question,
                    "true_answer": true_answer,
                    "predicted_answer": pred_answer,
                    "model": model_variant,
                    "confidence": float(getattr(resp, "confidence", 0.0)),
                    "answer_cosine_sim": float(ans_cos),
                    "exact_match": int(exact),
                    "type": q_type,
                    "level": q_level,
                    "retrieval_k": TOP_K,
                    "router_retrieval_ms": float(getattr(resp, "retrieval_ms", 0.0)),
                    "inference_ms": float(getattr(resp, "inference_ms", 0.0)),
                    "end_to_end_ms": float(getattr(resp, "end_to_end_ms", 0.0)),
                    "client_elapsed_ms": float(grpc_elapsed_ms),
                    "local_retrieval_ms": float(local_retrieval_ms),
                    "retr_cache_hit": bool(retr_cache_hit),
                    "question_hash": qid_hash,
                    "prompt_chars": len(question or ""),
                    "context_chars": len(ctx or ""),
                }
                results.append(row_out)

                log_event(
                    ts_ms=now_ms(),
                    event="qa",
                    collection=collection_name,
                    dataset=dataset,
                    story_id=story_id,
                    question_id=qid,
                    question_hash=qid_hash,
                    model=model_variant,
                    retrieval_k=TOP_K,
                    local_retrieval_ms=float(local_retrieval_ms),
                    retr_cache_hit=bool(retr_cache_hit),
                    prompt_chars=len(question or ""),
                    context_chars=len(ctx or ""),
                    answer_chars=len(pred_answer or ""),
                    grpc_elapsed_ms=float(grpc_elapsed_ms),
                    router_retrieval_ms=float(getattr(resp, "retrieval_ms", 0.0)),
                    inference_ms=float(getattr(resp, "inference_ms", 0.0)),
                    end_to_end_ms=float(getattr(resp, "end_to_end_ms", 0.0)),
                    total_ms=now_ms() - t_total0,
                    exact_match=int(exact),
                    answer_cosine_sim=float(ans_cos),
                )

            except grpc.RpcError as e:
                grpc_code = str(e.code())
                grpc_details = str(e.details())

                results.append(
                    {
                        "collection": collection_name,
                        "dataset": dataset,  # ✅ keep dataset even on error rows
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
                        "local_retrieval_ms": float(local_retrieval_ms),
                        "retr_cache_hit": bool(retr_cache_hit),
                        "question_hash": qid_hash,
                        "grpc_code": grpc_code,
                        "grpc_details": grpc_details,
                        "prompt_chars": len(question or ""),
                        "context_chars": len(ctx or ""),
                    }
                )

                log_event(
                    ts_ms=now_ms(),
                    event="qa_error",
                    collection=collection_name,
                    dataset=dataset,
                    story_id=story_id,
                    question_id=qid,
                    question_hash=qid_hash,
                    model=model_variant,
                    retrieval_k=TOP_K,
                    local_retrieval_ms=float(local_retrieval_ms),
                    retr_cache_hit=bool(retr_cache_hit),
                    prompt_chars=len(question or ""),
                    context_chars=len(ctx or ""),
                    grpc_code=grpc_code,
                    grpc_details=grpc_details,
                    total_ms=now_ms() - t_total0,
                )

            pbar.update(1)

    pbar.close()
    df_results = pd.DataFrame(results)
    print(f"Evaluation complete for '{collection_name}'. Rows: {len(df_results)}")
    return df_results


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
        f.write(f"JSON_LOGS: {JSON_LOGS}\n")
        f.write(f"RETR_CACHE_TTL_S: {RETR_CACHE_TTL_S}\n")
        f.write(f"TRIM_CHUNKS: {TRIM_CHUNKS}\n")
        f.write(f"MAX_CHARS_PER_CHUNK: {MAX_CHARS_PER_CHUNK}\n")
        f.write(f"Rows: {len(df_results)}\n\n")
    print(f"Saved summary: {summary_path}")


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

    if TRAIN_ROUTER_MODEL:
        import subprocess, sys
        print("Training router model from evaluation results...")
        env = os.environ.copy()
        env["ROUTER_MODEL_PATH"] = ROUTER_MODEL_PATH
        env["ROUTER_TRAIN_GLOB"] = ROUTER_TRAIN_GLOB

        r = subprocess.run([sys.executable, "/app/evaluate/train_router_model.py"], env=env)
        if r.returncode != 0:
            print("Router model training failed (non-fatal).")
        else:
            # ✅ upload to MinIO after successful training
            try:
                from s3_artifacts import upload_file
                if upload_file(ROUTER_MODEL_PATH):
                    print(f"Uploaded router model to MinIO: {ROUTER_MODEL_PATH}")
            except Exception as e:
                print(f"Upload failed (non-fatal): {e}")

    print("\n" + "=" * 70)
    print("All collection evaluations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
