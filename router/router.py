import asyncio
import itertools
import os
import grpc

import qa_pb2
import qa_pb2_grpc


def _split_backends(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


# Ollama-only variants (match your docker-compose env vars)
VARIANTS = {
    "llama3": _split_backends(os.getenv("BACKENDS_llama3", "")),
    "qwen2_5": _split_backends(os.getenv("BACKENDS_qwen2_5", "")),
    "mistral": _split_backends(os.getenv("BACKENDS_mistral", "")),
}

# remove any empty entries
VARIANTS = {k: v for k, v in VARIANTS.items() if v}


class RouterServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self):
        if not VARIANTS:
            raise RuntimeError(
                "No backends configured. Set BACKENDS_llama3 / BACKENDS_qwen2_5 / BACKENDS_mistral."
            )

        self.rr = {k: itertools.cycle(v) for k, v in VARIANTS.items()}

        # Safe default (or use DEFAULT_VARIANT env var)
        self.default_variant = os.getenv("DEFAULT_VARIANT", "llama3")
        if self.default_variant not in self.rr:
            self.default_variant = next(iter(self.rr.keys()))

        print(f"Router initialized with variants: {list(VARIANTS.keys())}")
        for variant, backends in VARIANTS.items():
            print(f"  • {variant}: {backends}")
        print(f"Default variant: {self.default_variant}")

    def _choose_variant(self, md: dict) -> str:
        # 1) Explicit model request wins
        requested = (md.get("variant") or "").strip()
        if requested:
            return requested

        # 2) Optional: dataset-based routing via env vars (no “policy” system)
        # Example env:
        #   DATASET_VARIANT_hotpotqa=llama3
        #   DATASET_VARIANT_narrativeqa=mistral
        dataset = (md.get("dataset") or "").strip().lower()
        if dataset:
            mapped = (os.getenv(f"DATASET_VARIANT_{dataset}") or "").strip()
            if mapped:
                return mapped

        # 3) Default
        return self.default_variant

    async def Answer(self, request, context):
        md = dict(context.invocation_metadata())
        variant = self._choose_variant(md)

        if variant not in self.rr:
            available = ", ".join(self.rr.keys())
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unknown variant '{variant}'. Available: {available}",
            )

        # lets clients verify which model was used
        context.set_trailing_metadata((("routed-variant", variant),))

        backends = VARIANTS[variant]
        last_err = None

        # try each backend once (round-robin order)
        for _ in range(len(backends)):
            backend = next(self.rr[variant])
            print(f"Routing [{variant}] -> {backend}")

            try:
                async with grpc.aio.insecure_channel(backend) as channel:
                    stub = qa_pb2_grpc.QAServerStub(channel)
                    return await stub.Answer(request)
            except grpc.aio.AioRpcError as e:
                last_err = e
                print(f"Backend failed {backend}: {e}")

        details = last_err.details() if last_err else "unknown error"
        context.abort(
            grpc.StatusCode.UNAVAILABLE,
            f"All backends for '{variant}' unavailable: {details}",
        )


async def serve():
    server = grpc.aio.server()
    qa_pb2_grpc.add_QAServerServicer_to_server(RouterServicer(), server)
    server.add_insecure_port("[::]:50050")
    await server.start()
    print("Router running on port 50050")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
