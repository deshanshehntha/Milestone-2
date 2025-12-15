import os
import time
import asyncio
import grpc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minio import Minio
import tempfile
import qa_pb2, qa_pb2_grpc


# -----------------------------
# Generation / prompt settings
# -----------------------------
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "1536"))   # total prompt tokens
MAX_NEW_TOKENS   = int(os.getenv("MAX_NEW_TOKENS", "64"))
MIN_NEW_TOKENS   = int(os.getenv("MIN_NEW_TOKENS", "1"))

DO_SAMPLE   = os.getenv("DO_SAMPLE", "0") == "1"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P       = float(os.getenv("TOP_P", "0.9"))

# Waiting for MinIO/model readiness
READY_OBJECT = os.getenv("MINIO_READY_OBJECT", "_READY")
READY_TIMEOUT_S = int(os.getenv("READY_TIMEOUT_S", "3600"))  # up to 1 hour
READY_POLL_S = int(os.getenv("READY_POLL_S", "5"))


def safe_model_max_len(tok) -> int:
    # Some tokenizers report absurdly large model_max_length (e.g. 1e30)
    m = getattr(tok, "model_max_length", None)
    if not m or m > 100_000:
        return 2048
    return int(m)


def make_minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


def wait_for_minio_ready(client: Minio, timeout_s: int = 180) -> None:
    """Wait until MinIO responds to an API call."""
    start = time.time()
    while True:
        try:
            client.list_buckets()
            print("✅ MinIO API is reachable", flush=True)
            return
        except Exception as e:
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Timed out waiting for MinIO API: {e}")
            print("⏳ Waiting for MinIO API...", flush=True)
            time.sleep(2)


def wait_for_models_ready(client: Minio, timeout_s: int = READY_TIMEOUT_S) -> None:
    """Wait until setup-minio writes models/_READY marker."""
    start = time.time()
    while True:
        try:
            client.stat_object("models", READY_OBJECT)
            print(f"✅ Found models/{READY_OBJECT} marker in MinIO", flush=True)
            return
        except Exception:
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Timed out waiting for models/{READY_OBJECT} in MinIO")
            print("⏳ Waiting for setup-minio to finish uploading models...", flush=True)
            time.sleep(READY_POLL_S)


class QAServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self, model_name, model_id):
        self.model_name = model_name
        self.model_id = model_id

        # Pick device + dtype (important for 2–3B models)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load model files from MinIO (blocks until ready marker exists)
        self.model_path = self.load_model_from_minio()

        # Load tokenizer/model from local path
        self.tok = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self.mdl = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.mdl.eval()
        self.mdl.to(self.device)

        # Ensure pad token exists
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id

        self.max_ctx = min(MAX_INPUT_TOKENS, safe_model_max_len(self.tok))

        print(
            f"[{self.model_name}] device={self.device} dtype={self.dtype} "
            f"max_ctx={self.max_ctx} max_new={MAX_NEW_TOKENS} do_sample={DO_SAMPLE}",
            flush=True
        )

    def load_model_from_minio(self):
        client = make_minio_client()

        # Wait for MinIO + the READY marker that indicates downloads/uploads are complete
        wait_for_minio_ready(client)
        wait_for_models_ready(client)

        bucket = "models"
        prefix = f"{self.model_name}/"
        tmp_dir = tempfile.mkdtemp()

        print(f" Loading model {self.model_name} from MinIO (bucket: {bucket}, prefix: {prefix})", flush=True)

        objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
        if not objects:
            raise Exception(
                f"No files found for {self.model_name} in MinIO bucket '{bucket}' with prefix '{prefix}'. "
                f"Check that setup-minio uploaded to models/{self.model_name}/"
            )

        print(f" Found {len(objects)} files in MinIO for {self.model_name}", flush=True)

        for obj in objects:
            filename = obj.object_name.replace(prefix, "")
            local_file = os.path.join(tmp_dir, filename)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            # Skip any marker files if they ever show up under the prefix
            if not filename:
                continue
            print(f"   Downloading {obj.object_name}", flush=True)
            client.fget_object(bucket, obj.object_name, local_file)

        print(f" Successfully loaded {self.model_name} from MinIO to: {tmp_dir}", flush=True)
        return tmp_dir

    def _build_prompt(self, question: str, context: str) -> str:
        system = (
            "You are a question-answering assistant. "
            "Answer the question using ONLY the provided context. "
            "Return ONLY the final answer (no explanation). "
            "If the answer is not present in the context, return 'unknown'."
        )

        # Prefer chat template for instruct/chat models
        if hasattr(self.tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            ]
            return self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Fallback for plain/base models
        return f"{system}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    def _truncate_context_to_fit(self, question: str, context: str) -> str:
        """
        Truncate *context* so we never truncate away the question/Answer marker.
        Strategy: Build prompt with empty context to measure overhead, then fit context.
        """
        empty_prompt = self._build_prompt(question=question, context="")
        overhead_ids = self.tok(empty_prompt, add_special_tokens=False).input_ids

        budget = self.max_ctx - len(overhead_ids)
        if budget <= 64:
            budget = 256

        ctx_ids = self.tok(context, add_special_tokens=False).input_ids
        if len(ctx_ids) <= budget:
            return context

        # Keep tail of context
        ctx_ids = ctx_ids[-budget:]
        return self.tok.decode(ctx_ids, skip_special_tokens=True)

    async def Answer(self, request, context):
        t0 = time.perf_counter()

        question = (request.question or "").strip()
        raw_context = (request.context or "").strip()

        trimmed_context = self._truncate_context_to_fit(question, raw_context)
        prompt = self._build_prompt(question, trimmed_context)

        # tokenize & move to device
        t_tok0 = time.perf_counter()
        inputs = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ctx,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        t_tok1 = time.perf_counter()

        # generate
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        if DO_SAMPLE:
            gen_kwargs["temperature"] = TEMPERATURE
            gen_kwargs["top_p"] = TOP_P

        with torch.no_grad():
            t_gen0 = time.perf_counter()
            outputs = self.mdl.generate(**inputs, **gen_kwargs)
            t_gen1 = time.perf_counter()

        # decode ONLY new tokens
        t_dec0 = time.perf_counter()
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        answer = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        answer = answer.split("\n")[0].strip()
        if not answer:
            answer = "unknown"

        confidence = 0.85 if answer != "unknown" else 0.15

        t_dec1 = time.perf_counter()

        tokenize_ms = (t_tok1 - t_tok0) * 1000.0
        forward_ms = (t_gen1 - t_gen0) * 1000.0
        decode_ms = (t_dec1 - t_dec0) * 1000.0
        e2e_ms = (time.perf_counter() - t0) * 1000.0

        print(
            f"[{self.model_name}] tokenize={tokenize_ms:.1f}ms "
            f"generate={forward_ms:.1f}ms decode={decode_ms:.1f}ms e2e={e2e_ms:.1f}ms",
            flush=True
        )
        print(f"[{self.model_name}] Q: {question[:80]} | A: {answer[:80]}", flush=True)

        return qa_pb2.AnswerResponse(
            answer=answer,
            confidence=float(confidence),
            retrieval_ms=float(tokenize_ms),
            inference_ms=float(forward_ms),
            end_to_end_ms=float(e2e_ms),
        )


async def serve():
    port = int(os.environ.get("PORT", "50051"))
    model_name = os.environ.get("MODEL_NAME", "qwen-1.5b")
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )
    qa_pb2_grpc.add_QAServerServicer_to_server(QAServicer(model_name, model_id), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f" Serving {model_name} ({model_id}) on port {port}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
