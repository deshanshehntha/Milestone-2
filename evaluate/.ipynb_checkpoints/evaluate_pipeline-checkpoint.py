import os
import time
import socket
from pathlib import Path

import grpc
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util

import qa_pb2, qa_pb2_grpc

# ---- NEW: Chroma imports ----
import chromadb
from chromadb.config import Settings


ROUTER_ADDR = "qa-router:50050"
VARIANTS = ["tinyroberta", "roberta_base", "bert_large"]

OUTPUT_PATH = "/mnt/results/all_model_results.csv"
SUMMARY_PATH = "/mnt/results/summary.txt"

DATASETS = {
    "Qasper": "/mnt/Qasper/processed_qasper_df.csv",
    "HotpotQA": "/mnt/HotpotQA/processed_hotpot_df.csv",
}

TOP_K = 4  # number of chunks to retrieve from Chroma
RETRIEVE_TIMEOUT_S = 5

os.makedirs("/mnt/results", exist_ok=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def wait_for_datasets():
    print("Waiting for datasets to be ready...")
    max_wait = 200
    start_time = time.time()
    required = [Path(DATASETS["Qasper"]), Path(DATASETS["HotpotQA"])]
    while True:
        if all(f.exists() for f in required):
            print("Both datasets found.")
            return True
        if time.time() - start_time > max_wait:
            print("Timeout waiting for datasets.")
            return False
        print("...still waiting for datasets...")
        time.sleep(10)


def wait_for_router(addr="qa-router", port=50050, timeout=300):
    print(f" Waiting for router at {addr}:{port}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((addr, port), timeout=5):
                print("Router is ready.")
                return True
        except OSError:
            time.sleep(5)
    print("Router not reachable after timeout.")
    return False



def cosine_similarity(prediction, ground_truth):
    if not prediction or not ground_truth:
        return 0.0
    emb_pred = embedder.encode(prediction, convert_to_tensor=True)
    emb_gt = embedder.encode(ground_truth, convert_to_tensor=True)
    return float(util.cos_sim(emb_pred, emb_gt))


def ask_question(stub, question, paper_id, variant, context_text="", retries=3):
    metadata = (("variant", variant),)
    req = qa_pb2.Question(
        question=question,
        context=context_text or "",
        paper_id=str(paper_id or "")
    )

    for attempt in range(retries):
        try:
            start = time.time()
            resp = stub.Answer(req, metadata=metadata)
            elapsed = (time.time() - start) * 1000
            return {
                "variant": variant,
                "question": question,
                "paper_id": paper_id,
                "answer": resp.answer,
                "confidence": resp.confidence,
                "retrieval_ms": resp.retrieval_ms,
                "inference_ms": resp.inference_ms,
                "end_to_end_ms": resp.end_to_end_ms,
                "client_latency_ms": elapsed,
            }
        except grpc.RpcError as e:
            print(f" gRPC error ({variant}) attempt {attempt+1}: {e.code().name}")
            time.sleep(5)
    print(f" Failed after {retries} retries for {variant}.")
    return {
        "variant": variant,
        "question": question,
        "paper_id": paper_id,
        "answer": "",
        "confidence": 0.0,
        "retrieval_ms": 0.0,
        "inference_ms": 0.0,
        "end_to_end_ms": 0.0,
        "client_latency_ms": 0.0,
    }



def connect_chroma():
    # Uses your docker service name "chroma" and exposed port 8000
    for attempt in range(10):
        try:
            client = chromadb.HttpClient(
                host="chroma",
                port=8000,
                settings=Settings(allow_reset=False)
            )
            # simple ping: list collections (won't throw if reachable)
            _ = client.list_collections()
            print("Connected to Chroma server.")
            return client
        except Exception as e:
            print(f" Waiting for Chroma to be ready (attempt {attempt+1}/10)... {e}")
            time.sleep(3)
    raise ConnectionError("Could not connect to Chroma server after 10 attempts.")


def load_collections(chroma_client):
    # Must match names used during ingestion
    colls = {}
    try:
        colls["Qasper"]   = chroma_client.get_collection("qasper_data_collection")
    except Exception as e:
        print(f"⚠️ Could not open Qasper collection: {e}")
    try:
        colls["HotpotQA"] = chroma_client.get_collection("hotpotqa_data_collection")
    except Exception as e:
        print(f"⚠️ Could not open HotpotQA collection: {e}")
    return colls


def retrieve_context(collection, dataset_name, question_text, paper_or_id=None, top_k=TOP_K):
    """
    Query Chroma for relevant chunks. If an id is provided, try to filter to that doc
    (paper_id for Qasper; hotpot_id for HotpotQA). Falls back to unfiltered query if needed.
    """
    if collection is None:
        return ""

    # Build an optional filter depending on dataset
    where = None
    if paper_or_id:
        if dataset_name == "Qasper":
            where = {"paper_id": str(paper_or_id)}
        elif dataset_name == "HotpotQA":
            where = {"hotpot_id": str(paper_or_id)}

    # First: try filtered search (if where provided)
    try:
        if where:
            res = collection.query(query_texts=[question_text], n_results=top_k, where=where)
        else:
            res = collection.query(query_texts=[question_text], n_results=top_k)

        docs = res.get("documents", [[]])[0]
        # Dedup while preserving order
        seen, unique_docs = set(), []
        for d in docs:
            if d not in seen:
                unique_docs.append(d)
                seen.add(d)

        return "\n\n".join(unique_docs)
    except Exception as e:
        print(f"Retrieval failed (primary), falling back without filters: {e}")

    # Fallback: unfiltered query
    try:
        res = collection.query(query_texts=[question_text], n_results=top_k)
        docs = res.get("documents", [[]])[0]
        seen, unique_docs = set(), []
        for d in docs:
            if d not in seen:
                unique_docs.append(d)
                seen.add(d)
        return "\n\n".join(unique_docs)
    except Exception as e:
        print(f"Retrieval failed (fallback): {e}")
        return ""



def evaluate_dataset(name, path, stub, collections):
    print(f" Evaluating dataset: {name}")
    df = pd.read_csv(path)

    if "question" not in df.columns or len(df) == 0:
        print(f"No valid questions found in {name}")
        return None

    # Normalize answers
    if "free_form_answer" in df.columns:
        df["answer_text"] = df["free_form_answer"]
    elif "answer" in df.columns:
        df["answer_text"] = df["answer"]
    else:
        df["answer_text"] = ""

    # Determine row id field for retrieval filter
    id_field = None
    if name == "Qasper":
        id_field = "paper_id" if "paper_id" in df.columns else None
    elif name == "HotpotQA":
        # ingestion used "hotpot_id" as metadata, but row carries "id" in CSV
        id_field = "id" if "id" in df.columns else None

    results = []
    chroma_coll = collections.get(name)

    for idx, row in df.head(10).iterrows():  # limit for test run
        q = str(row["question"])
        gt = str(row["answer_text"])
        pid = str(row[id_field]) if id_field and id_field in row else None

        # --- NEW: dynamic retrieval from Chroma ---
        ctx = retrieve_context(chroma_coll, name, q, paper_or_id=pid, top_k=TOP_K)

        # If retrieval fails, optionally fall back to any CSV 'context' column if present
        if not ctx:
            ctx = str(row.get("context", "")) if "context" in df.columns else ""

        for variant in VARIANTS:
            print(f"→ [{variant}] {q[:80]}...")
            model_out = ask_question(stub, q, pid, variant, context_text=ctx)
            model_out["dataset"] = name
            model_out["ground_truth"] = gt
            model_out["cosine_sim"] = cosine_similarity(model_out["answer"], gt)
            model_out["overhead_ms"] = max(
                0.0,
                model_out["end_to_end_ms"] - model_out["inference_ms"] - model_out["retrieval_ms"]
            )
            results.append(model_out)

    return pd.DataFrame(results)



def main():
    if not wait_for_datasets():
        return
    if not wait_for_router():
        return

    chroma_client = connect_chroma()
    collections = load_collections(chroma_client)

    all_results = []
    with grpc.insecure_channel(ROUTER_ADDR) as channel:
        stub = qa_pb2_grpc.QAServerStub(channel)
        for name, path in DATASETS.items():
            if os.path.exists(path):
                res_df = evaluate_dataset(name, path, stub, collections)
                if res_df is not None:
                    all_results.append(res_df)

    if not all_results:
        print(" No results generated.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    summary = (
        combined.groupby("variant")[["retrieval_ms", "inference_ms", "end_to_end_ms", "overhead_ms"]]
        .mean()
        .round(2)
    )
    summary.to_csv("/mnt/results/latency_breakdown_by_variant.csv")
    print(summary)

    avg_sim = combined["cosine_sim"].mean()
    avg_conf = combined["confidence"].mean()
    avg_latency = combined["end_to_end_ms"].mean()

    print(f"\n Saved all results to {OUTPUT_PATH}")
    print("Average Cosine Similarity:", avg_sim)
    print("Average Confidence:", avg_conf)
    print("Average Latency (ms):", avg_latency)

    with open(SUMMARY_PATH, "w") as f:
        f.write(f"Average Cosine Similarity: {avg_sim}\n")
        f.write(f"Average Confidence: {avg_conf}\n")
        f.write(f"Average Latency (ms): {avg_latency}\n")

    print(f" Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
