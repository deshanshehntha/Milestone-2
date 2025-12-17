#!/usr/bin/env python3
import os
import glob
import joblib
import pandas as pd

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


def main():
    results_glob = os.getenv("ROUTER_TRAIN_GLOB", "/data/results/**/evaluation_results.csv")
    out_path = os.getenv("ROUTER_MODEL_PATH", "/data/router_model/router_model.joblib")

    csvs = sorted(glob.glob(results_glob, recursive=True))
    if not csvs:
        raise SystemExit(f"No CSVs found for ROUTER_TRAIN_GLOB={results_glob}")

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    # --- normalize columns / make dataset always present ---
    if "dataset" not in df.columns:
        df["dataset"] = ""
    if "collection" not in df.columns:
        df["collection"] = "unknown"

    df["dataset"] = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    df["collection"] = df["collection"].fillna("").astype(str).str.strip().str.lower()

    # fallback: dataset := collection if empty
    df.loc[df["dataset"] == "", "dataset"] = df.loc[df["dataset"] == "", "collection"]
    df.loc[df["dataset"] == "", "dataset"] = "unknown"

    # ensure feature columns exist
    for c in ["prompt_chars", "context_chars", "retrieval_k"]:
        if c not in df.columns:
            df[c] = 0

    # must have these to train
    required = ["question_id", "model", "answer_cosine_sim", "exact_match"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in eval CSV(s): {missing}")

    # group key (prefer question_id)
    key = "question_id"

    # labels: best model per question
    y = df.groupby(key).apply(pick_best_model)

    # features: first row per question
    first = df.groupby(key).first().copy()
    X = first[["dataset", "prompt_chars", "context_chars", "retrieval_k"]].copy()

    # build model
    cat = ["dataset"]
    num = ["prompt_chars", "context_chars", "retrieval_k"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )

    clf = LogisticRegression(max_iter=500, n_jobs=1, multi_class="auto")

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"model": pipe}, out_path)
    print(f"Saved router model: {out_path}")


if __name__ == "__main__":
    main()
