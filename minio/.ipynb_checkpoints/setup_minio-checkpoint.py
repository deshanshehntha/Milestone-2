#!/usr/bin/env python3
import os
import time
import io
import torch
from minio import Minio
from minio.error import S3Error
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}



MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

DOWNLOAD_DIR = "/downloaded_models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def wait_for_minio(client: Minio, timeout_s: int = 120):
    start = time.time()
    while True:
        try:
            client.list_buckets()
            print(" MinIO is ready", flush=True)
            return
        except Exception as e:
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Timed out waiting for MinIO: {e}")
            print(" Waiting for MinIO...", flush=True)
            time.sleep(2)

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

wait_for_minio(client)

for bucket in ["models", "results", "logs"]:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"Created bucket: {bucket}", flush=True)

def upload_dir_to_minio(local_dir: str, bucket: str, prefix: str):
    for root, _, files in os.walk(local_dir):
        for f in files:
            local_path = os.path.join(root, f)
            rel_path = os.path.relpath(local_path, local_dir)
            remote_path = f"{prefix}/{rel_path}".replace("\\", "/")
            client.fput_object(bucket, remote_path, local_path)

for name, model_id in MODELS.items():
    local_path = os.path.join(DOWNLOAD_DIR, name)
    os.makedirs(local_path, exist_ok=True)

    if not os.path.exists(os.path.join(local_path, "config.json")):
        print(f"Downloading {model_id} ...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
    else:
        print(f"Found cached model files for {name}, skipping download.", flush=True)

    print(f"Uploading {name} to MinIO bucket 'models' ...", flush=True)
    upload_dir_to_minio(local_path, "models", name)
    print(f"Uploaded {name}", flush=True)

client.put_object(
    "models",
    "_READY",
    data=io.BytesIO(b"ok"),
    length=2,
    content_type="text/plain",
)
print("All models uploaded to MinIO successfully. READY marker written.", flush=True)
