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
        self.model_path = self.load_model()
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.mdl = AutoModelForQuestionAnswering.from_pretrained(self.model_path).eval()

    def load_model(self):
        """Try to fetch model from MinIO; fallback to Hugging Face."""
        endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)
        tmp_dir = tempfile.mkdtemp()
        prefix = f"models/{self.model_name}/"

        print(f" Trying to download {self.model_name} from MinIO...")

        try:
            objects = list(client.list_objects("models", prefix=prefix, recursive=True))
            if not objects:
                raise S3Error("No files found in MinIO bucket for this model.")

            for obj in objects:
                local_file = os.path.join(tmp_dir, os.path.basename(obj.object_name))
                client.fget_object("models", obj.object_name, local_file)

            print(f" Loaded {self.model_name} from MinIO: {tmp_dir}")
            return tmp_dir

        except Exception as e:
            print(f"Could not load {self.model_name} from MinIO: {e}")
            print(f"Downloading {self.model_id} from Hugging Face instead...")
            return self.model_id

    async def Answer(self, request, context):
        t0 = time.perf_counter()
        ins = self.tok(request.question, return_tensors="pt", truncation=True, max_length=384)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.mdl(**ins)
        si = int(out.start_logits.argmax()); ei = int(out.end_logits.argmax())
        ans = self.tok.decode(ins.input_ids[0, si:ei+1], skip_special_tokens=True)
        inf_ms = (time.perf_counter()-t1)*1000
        e2e_ms = (time.perf_counter()-t0)*1000
        conf = float(out.start_logits.softmax(-1)[0,si] * out.end_logits.softmax(-1)[0,ei])
        return qa_pb2.AnswerResponse(
            answer=ans,
            confidence=conf,
            retrieval_ms=0.0,
            inference_ms=inf_ms,
            end_to_end_ms=e2e_ms
        )


async def serve():
    port = int(os.environ.get("PORT", "50051"))
    model_name = os.environ.get("MODEL_NAME", "tinyroberta")
    model_id = os.environ.get("MODEL_ID", "deepset/tinyroberta-squad2")

    server = grpc.aio.server()
    qa_pb2_grpc.add_QAServerServicer_to_server(QAServicer(model_name, model_id), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"Serving {model_name} on :{port}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
