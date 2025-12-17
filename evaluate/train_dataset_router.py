import os, glob
import joblib
import pandas as pd

from s3_artifacts import upload_file


def load_df(results_glob: str) -> pd.DataFrame:
    csvs = sorted(glob.glob(results_glob, recursive=True))
    if not csvs:
        raise SystemExit(f"No CSVs found for ROUTER_TRAIN_GLOB={results_glob}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    # dataset/collection normalization
    if "dataset" not in df.columns:
        df["dataset"] = ""
    if "collection" not in df.columns:
        df["collection"] = "unknown"

    df["dataset"] = df["dataset"].fillna("").astype(str).str.strip().str.lower()
    df["collection"] = df["collection"].fillna("unknown").astype(str).str.strip().str.lower()
    df.loc[df["dataset"] == "", "dataset"] = df.loc[df["dataset"] == "", "collection"]
    df.loc[df["dataset"] == "", "dataset"] = "unknown"

    # model
    if "model" not in df.columns:
        raise SystemExit("Missing required column: model")
    df["model"] = df["model"].fillna("unknown").astype(str).str.strip().str.lower()

    # metrics
    if "exact_match" not in df.columns:
        raise SystemExit("Missing required column: exact_match")
    df["exact_match"] = pd.to_numeric(df["exact_match"], errors="coerce").fillna(0).astype(float)

    if "answer_cosine_sim" not in df.columns:
        df["answer_cosine_sim"] = 0.0
    df["answer_cosine_sim"] = pd.to_numeric(df["answer_cosine_sim"], errors="coerce").fillna(0).astype(float)

    return df


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


def main():
    results_glob = os.getenv("ROUTER_TRAIN_GLOB", "/data/results/**/evaluation_results.csv")
    out_path = os.getenv("ROUTER_MODEL_PATH", "/data/router_model/router_model.joblib")

    em_w = float(os.getenv("ROUTER_EM_WEIGHT", "0.25"))
    cos_w = float(os.getenv("ROUTER_COS_WEIGHT", "1.0"))

    df = load_df(results_glob)

    # dataset x model leaderboard
    leaderboard = (
        df.groupby(["dataset", "model"], as_index=False)
          .agg(
              mean_em=("exact_match", "mean"),
              mean_cos=("answer_cosine_sim", "mean")
          )
    )

    winners = (
        leaderboard.sort_values(
            ["dataset", "mean_em", "mean_cos", "model"],
            ascending=[True, False, False, True],
        )
        .groupby("dataset", as_index=False)
        .first()
    )

    mapping = dict(zip(winners["dataset"].tolist(), winners["model"].tolist()))

    obj = {
        "router_type": "dataset",
        "dataset_router": mapping,
        "meta": {
            "em_weight": em_w,
            "cos_weight": cos_w,
            "uses_latency": False,
            "rows": int(len(df)),
            "datasets": int(df["dataset"].nunique()),
            "models": sorted(df["model"].unique().tolist()),
        },
        # optional debug info
        "dataset_leaderboard": winners[
            ["dataset", "model", "mean_em", "mean_cos"]
        ].to_dict("records"),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(obj, out_path)
    print(f"Saved DATASET router artifact: {out_path}")
    print("Dataset mapping:", mapping)

    maybe_upload(out_path)


if __name__ == "__main__":
    main()
