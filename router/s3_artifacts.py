import os
import boto3
from botocore.client import Config


def _client():
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip() or None
    access = os.getenv("S3_ACCESS_KEY", "").strip() or None
    secret = os.getenv("S3_SECRET_KEY", "").strip() or None
    region = os.getenv("S3_REGION", "us-east-1").strip() or "us-east-1"

    # MinIO wants path-style addressing commonly
    cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=region,
        config=cfg,
    )


def upload_file(local_path: str, bucket: str, key: str):
    c = _client()
    c.upload_file(local_path, bucket, key)


def download_file(local_path: str, bucket: str, key: str):
    c = _client()
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    c.download_file(bucket, key, local_path)
