import os
import time
import asyncio
import grpc
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from minio import Minio
from minio.error import S3Error
import tempfile
import qa_pb2, qa_pb2_grpc


class QAServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self, model_name, model_id):
        self.model_name = model_name
        self.model_id = model_id
        self.model_path = self.load_or_fetch_model()
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.mdl = AutoModelForQuestionAnswering.from_pretrained(self.model_path).eval()

    def load_or_fetch_model(self):
        """Try to load model from MinIO, else download from Hugging Face and upload back."""
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

        bucket = "models"
        prefix = f"models/{self.model_name}/"
        tmp_dir = tempfile.mkdtemp()
        print(f" Checking MinIO for model: {prefix}")

        try:
            objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
            if objects:
                for obj in objects:
                    local_file = os.path.join(tmp_dir, os.path.basename(obj.object_name))
                    client.fget_object(bucket, obj.object_name, local_file)
                print(f" Loaded {self.model_name} from MinIO cache: {tmp_dir}")
                return tmp_dir
            else:
                print(f" No files found for {self.model_name} in MinIO. Fetching from Hugging Face...")

        except Exception as e:
            print(f" MinIO unavailable or model not found: {e}")
            print(f" Downloading {self.model_id} from Hugging Face...")

  
        model_dir = tempfile.mkdtemp()
        mdl = AutoModelForQuestionAnswering.from_pretrained(self.model_id)
        tok = AutoTokenizer.from_pretrained(self.model_id)
        mdl.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)

        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
            for root, _, files in os.walk(model_dir):
                for f in files:
                    local = os.path.join(root, f)
                    remote = f"models/{self.model_name}/{f}"
                    client.fput_object(bucket, remote, local)
            print(f" Uploaded {self.model_name} to MinIO for future use.")
        except Exception as e:
            print(f" Could not upload {self.model_name} to MinIO: {e}")

        return model_dir

    # async def Answer(self, request, context):
    #     t0 = time.perf_counter()
    #     # ins = self.tok(request.question, return_tensors="pt", truncation=True, max_length=384)
    #     ins = self.tok(
    #         request.question,
    #         request.context or "",      
    #         return_tensors="pt",
    #         truncation=True,
    #         max_length=384
    #     )
    #     t1 = time.perf_counter()
    #     with torch.no_grad():
    #         out = self.mdl(**ins)
    #     si, ei = int(out.start_logits.argmax()), int(out.end_logits.argmax())
    #     ans = self.tok.decode(ins.input_ids[0, si:ei+1], skip_special_tokens=True)
    #     inf_ms = (time.perf_counter() - t1) * 1000
    #     e2e_ms = (time.perf_counter() - t0) * 1000
    #     conf = float(out.start_logits.softmax(-1)[0, si] * out.end_logits.softmax(-1)[0, ei])
    #     print(f"[{self.model_name}] Received Q: {request.question[:80]}...")
    #     print(f"[{self.model_name}] Context snippet: {request.context[:120]}...")
    #     return qa_pb2.AnswerResponse(
    #         answer=ans,
    #         confidence=conf,
    #         retrieval_ms=0.0,
    #         inference_ms=inf_ms,
    #         end_to_end_ms=e2e_ms,
    #     )

    async def Answer(self, request, context):
            import time
            t0 = time.perf_counter()
    
            # --- tokenize ---
            t_tok0 = time.perf_counter()
            ins = self.tok(
                request.question,
                request.context or "",            # <-- ensure context is included
                return_tensors="pt",
                truncation=True,
                max_length=384,
                padding="max_length"
            )
            t_tok1 = time.perf_counter()
    
            # --- forward ---
            with torch.no_grad():
                t_fwd0 = time.perf_counter()
                out = self.mdl(**ins)
                t_fwd1 = time.perf_counter()
    
            # --- decode ---
            t_dec0 = time.perf_counter()
            si = int(out.start_logits.argmax())
            ei = int(out.end_logits.argmax())
            ans = self.tok.decode(ins.input_ids[0, si:ei+1], skip_special_tokens=True)
            # confidence = P(start=si)*P(end=ei)
            conf = float(out.start_logits.softmax(-1)[0, si] * out.end_logits.softmax(-1)[0, ei])
            t_dec1 = time.perf_counter()
    
            # timings
            tokenize_ms = (t_tok1 - t_tok0) * 1000.0
            forward_ms  = (t_fwd1 - t_fwd0) * 1000.0
            decode_ms   = (t_dec1 - t_dec0) * 1000.0
            e2e_ms      = (time.perf_counter() - t0) * 1000.0
    
            print(f"[{self.model_name}] tokenize={tokenize_ms:.1f}ms forward={forward_ms:.1f}ms decode={decode_ms:.1f}ms e2e={e2e_ms:.1f}ms", flush=True)
    
            return qa_pb2.AnswerResponse(
                answer=ans,
                confidence=conf,
                retrieval_ms=tokenize_ms,  # repurpose field for tokenize time
                inference_ms=forward_ms,
                end_to_end_ms=e2e_ms,
            )

async def serve():
    port = int(os.environ.get("PORT", "50051"))
    model_name = os.environ.get("MODEL_NAME", "tinyroberta")
    model_id = os.environ.get("MODEL_ID", "deepset/tinyroberta-squad2")

    server = grpc.aio.server()
    qa_pb2_grpc.add_QAServerServicer_to_server(QAServicer(model_name, model_id), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f" Serving {model_name} ({model_id}) on port {port}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
