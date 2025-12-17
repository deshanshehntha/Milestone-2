#!/usr/bin/env python3
import os
import glob
import joblib
import pandas as pd
from s3_artifacts import upload_file

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


def pick_best_model(group: pd.DataFrame) -> str:
    em_w = float(os.getenv("ROUTER_EM_WEIGHT", "0.25"))
    lat_w = float(os.getenv("ROUTER_LAT_WEIGHT", "0.0002"))

    inf_ms = group.get("inference_ms", pd.Series([0.0] * len(group))).fillna(
        group.get("client_elapsed_ms", pd.Series([0.0] * len(group))).fillna(0.0)
    )

    utility = (
        group.get("answer_cosine_sim", 0.0).fillna(0.0)
        + em_w * group.get("exact_match", 0.0).fillna(0.0)
        - lat_w * inf_ms
    )
    best_idx = utility.idxmax()
    return str(group.loc[best_idx, "model"])

def maybe_upload(out_path: str):
    bucket = os.getenv("S3_BUCKET", "").strip()
    key = os.getenv("S3_KEY", "").strip()
    enabled = os.getenv("ROUTER_MODEL_S3_UPLOAD", "1").strip()  # default ON

    if enabled != "1":
        print("S3 upload disabled via ROUTER_MODEL_S3_UPLOAD!=1")
        return
    if not bucket or not key:
        print("S3_BUCKET/S3_KEY not set; skipping upload")
        return

    upload_file(out_path, bucket, key)
    print(f"Uploaded router artifact to s3://{bucket}/{key}")

    
def main() -> None:
    results_glob = os.getenv("ROUTER_TRAIN_GLOB", "/data/results/**/evaluation_results.csv")
    out_path = os.getenv("ROUTER_MODEL_PATH", "/data/router_model/router_model.joblib")

    csvs = sorted(glob.glob(results_glob, recursive=True))
    if not csvs:
        raise SystemExit(f"No CSVs found for ROUTER_TRAIN_GLOB={results_glob}")

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    # Normalize dataset/collection
    if "dataset" not in df.columns:
        df["dataset"] = ""
    if "collection" not in df.columns:
        df["collection"] = "unknown"

    df["dataset"] = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    df["collection"] = df["collection"].fillna("").astype(str).str.strip().str.lower()
    df.loc[df["dataset"] == "", "dataset"] = df.loc[df["dataset"] == "", "collection"]
    df.loc[df["dataset"] == "", "dataset"] = "unknown"

    # Ensure feature cols exist (REMOVED retrieval_k)
    for c in ["prompt_chars", "context_chars"]:
        if c not in df.columns:
            df[c] = 0

    required = ["question_id", "model", "answer_cosine_sim", "exact_match"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in eval CSV(s): {missing}")

    key = "question_id"

    # Labels: best model per question
    try:
        y = df.groupby(key, include_groups=False).apply(pick_best_model)
    except TypeError:
        y = df.groupby(key).apply(pick_best_model)

    # Features at question-level (REMOVED retrieval_k)
    first = df.groupby(key).first().copy()
    X = first[["dataset", "prompt_chars", "context_chars"]].copy()
    X = X.loc[y.index]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["dataset"]),
            ("num", "passthrough", ["prompt_chars", "context_chars"]),
        ]
    )

    # Added class_weight="balanced" (to avoid always predicting the majority model)
    clf = LogisticRegression(max_iter=10000, class_weight="balanced")
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    obj = {
        "router_type": "ml",
        "ml_model": pipe,
        "meta": {
            "em_weight": float(os.getenv("ROUTER_EM_WEIGHT", "0.25")),
            "lat_weight": float(os.getenv("ROUTER_LAT_WEIGHT", "0.0002")),
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(obj, out_path)
    print(f"Saved ML router artifact: {out_path}")

    maybe_upload(out_path)

if __name__ == "__main__":
    main()
