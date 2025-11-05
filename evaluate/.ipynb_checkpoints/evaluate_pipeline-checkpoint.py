import grpc
import pandas as pd
import numpy as np
import time
import socket
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import qa_pb2, qa_pb2_grpc

ROUTER_ADDR = "qa-router:50050"  
VARIANTS = ["tinyroberta", "roberta_base", "bert_large"]
OUTPUT_PATH = "/mnt/results/all_model_results.csv"
SUMMARY_PATH = "/mnt/results/summary.txt"

DATASETS = {
    "Qasper": "/mnt/Qasper/processed_qasper_df.csv",
    "HotpotQA": "/mnt/HotpotQA/processed_hotpot_df.csv",
}

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


def evaluate_dataset(name, path, stub):
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
    
    # Normalize context
    if "context" in df.columns:
        df["context"] = df["context"]
    elif "context" in df.columns:
        df["context"] = df["context"]
    else:
        df["context"] = ""

    results = []
    for idx, row in df.head(10).iterrows():  # limit for test run
        q = str(row["question"])
        gt = str(row["answer_text"])
        pid = row["id"] if "id" in row else row.get("paper_id", None)
        ctx = str(row.get("context", ""))
        
        for variant in VARIANTS:
            print(f"→ [{variant}] {q[:80]}...")
            model_out = ask_question(stub, q, pid, variant, context_text=ctx)
            model_out["dataset"] = name
            model_out["ground_truth"] = gt
            model_out["cosine_sim"] = cosine_similarity(model_out["answer"], gt)
            model_out["overhead_ms"] = max(0.0,
            model_out["end_to_end_ms"] - model_out["inference_ms"] - model_out["retrieval_ms"]
            )
            results.append(model_out)
    return pd.DataFrame(results)


def main():
    if not wait_for_datasets():
        return
    if not wait_for_router():
        return

    all_results = []
    with grpc.insecure_channel(ROUTER_ADDR) as channel:
        stub = qa_pb2_grpc.QAServerStub(channel)
        for name, path in DATASETS.items():
            if os.path.exists(path):
                res_df = evaluate_dataset(name, path, stub)
                if res_df is not None:
                    all_results.append(res_df)

    if not all_results:
        print(" No results generated.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    summary = combined.groupby("variant")[["retrieval_ms","inference_ms","end_to_end_ms","overhead_ms"]].mean().round(2)
    summary.to_csv("/mnt/results/latency_breakdown_by_variant.csv")
    print(summary)


    avg_sim = combined["cosine_sim"].mean()
    avg_conf = combined["confidence"].mean()
    avg_latency = combined["end_to_end_ms"].mean()

    print(f"\n Saved all results to {OUTPUT_PATH}")
    print("Average Cosine Similarity:", avg_sim)
    print("Average Confidence:", avg_conf)
    print("Average Latency (ms):", avg_latency)

    # Save quick summary to text file
    with open(SUMMARY_PATH, "w") as f:
        f.write(f"Average Cosine Similarity: {avg_sim}\n")
        f.write(f"Average Confidence: {avg_conf}\n")
        f.write(f"Average Latency (ms): {avg_latency}\n")

    print(f" Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
