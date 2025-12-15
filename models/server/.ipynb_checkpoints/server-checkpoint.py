import os
import time
import asyncio
import grpc
import json
import urllib.request
import urllib.error
import qa_pb2, qa_pb2_grpc


# -----------------------------
# Ollama settings
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# prompt/limits
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))   # simple safe cap
MAX_NEW_TOKENS    = int(os.getenv("MAX_NEW_TOKENS", "128"))

DO_SAMPLE   = os.getenv("DO_SAMPLE", "0") == "1"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P       = float(os.getenv("TOP_P", "0.9"))

# basic timeouts
HTTP_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "120"))


def _truncate_context(context: str) -> str:
    context = (context or "").strip()
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    # keep the tail (often contains the most relevant chunk in your concat format)
    return context[-MAX_CONTEXT_CHARS:]


def _build_prompt(question: str, context: str) -> str:
    system = (
        "You are a question-answering assistant. "
        "Answer the question using ONLY the provided context. "
        "Return ONLY the final answer (no explanation). "
        "If the answer is not present in the context, return 'unknown'."
    )

    # Ollama supports plain prompts reliably across models.
    # (Some Ollama models also support chat, but prompt is simplest + consistent.)
    return (
        f"{system}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )


def _ollama_generate(prompt: str) -> str:
    """
    Calls Ollama /api/generate (non-streaming) and returns response text.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            # NOTE: num_predict is the primary generation length knob in Ollama
            "num_predict": MAX_NEW_TOKENS,
        },
    }

    if DO_SAMPLE:
        payload["options"]["temperature"] = TEMPERATURE
        payload["options"]["top_p"] = TOP_P
    else:
        # greedy-ish (Ollama will still have some default sampling unless controlled)
        payload["options"]["temperature"] = 0.0

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8")
            obj = json.loads(body)
            return (obj.get("response") or "").strip()
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError(f"Ollama HTTPError {e.code}: {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e


class QAServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"[{self.model_name}] using OLLAMA_HOST={OLLAMA_HOST} OLLAMA_MODEL={OLLAMA_MODEL}", flush=True)

    async def Answer(self, request, context):
        t0 = time.perf_counter()

        question = (request.question or "").strip()
        raw_context = (request.context or "").strip()

        # "retrieval_ms" in your old server tracked tokenize time;
        # here we treat it as "prompt prep" time to keep your schema stable.
        t_prep0 = time.perf_counter()
        trimmed_context = _truncate_context(raw_context)
        prompt = _build_prompt(question, trimmed_context)
        t_prep1 = time.perf_counter()

        # Ollama inference
        t_inf0 = time.perf_counter()
        try:
            answer = _ollama_generate(prompt)
        except Exception as e:
            # return unknown but also log error
            print(f"[{self.model_name}] Ollama error: {e}", flush=True)
            answer = "unknown"
        t_inf1 = time.perf_counter()

        # clean answer similar to your HF server behavior
        answer = (answer or "").strip()
        answer = answer.split("\n")[0].strip()
        if not answer:
            answer = "unknown"

        # confidence: Ollama doesn't provide it. Keep your simple heuristic.
        confidence = 0.85 if answer != "unknown" else 0.15

        prep_ms = (t_prep1 - t_prep0) * 1000.0
        inf_ms  = (t_inf1 - t_inf0) * 1000.0
        e2e_ms  = (time.perf_counter() - t0) * 1000.0

        print(
            f"[{self.model_name}] prep={prep_ms:.1f}ms "
            f"ollama_infer={inf_ms:.1f}ms e2e={e2e_ms:.1f}ms | "
            f"Q: {question[:60]} | A: {answer[:60]}",
            flush=True
        )

        return qa_pb2.AnswerResponse(
            answer=answer,
            confidence=float(confidence),
            retrieval_ms=float(prep_ms),
            inference_ms=float(inf_ms),
            end_to_end_ms=float(e2e_ms),
        )


async def serve():
    port = int(os.environ.get("PORT", "50051"))
    model_name = os.environ.get("MODEL_NAME", "ollama-model")

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )
    qa_pb2_grpc.add_QAServerServicer_to_server(QAServicer(model_name), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"Serving {model_name} (OLLAMA_MODEL={OLLAMA_MODEL}) on port {port}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
