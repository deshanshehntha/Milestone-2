import os, time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from minio import Minio
from minio.error import S3Error

MODELS = {
    "tinyroberta": "deepset/tinyroberta-squad2",
    "roberta-base": "deepset/roberta-base-squad2",
    "bert-large": "deepset/bert-large-uncased-whole-word-masking-finetuned-squad2"
}

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

print(" Waiting for MinIO to start...")
time.sleep(10)

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# Ensure required buckets exist
for bucket in ["models", "results", "logs"]:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f" Created bucket: {bucket}")

os.makedirs("/downloaded_models", exist_ok=True)

# Download and upload models
for name, model_id in MODELS.items():
    print(f"Downloading {model_id} ...")
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    path = f"/downloaded_models/{name}"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)

    print(f"Uploading {name} to MinIO ...")
    for root, _, files in os.walk(path):
        for f in files:
            local = os.path.join(root, f)
            remote = f"models/{name}/{f}"
            client.fput_object("models", remote, local)
    print(f"Uploaded {name}")

print("All models uploaded to MinIO successfully.")
