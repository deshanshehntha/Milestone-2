import asyncio
import itertools
import os
import json
import grpc

import qa_pb2
import qa_pb2_grpc

from model_router import pick_backend


def _split_backends(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _safe_int(x, default: int):
    try:
        return int(x)
    except Exception:
        return default


VARIANTS = {
    "llama3": _split_backends(os.getenv("BACKENDS_llama3", "")),
    "qwen2_5": _split_backends(os.getenv("BACKENDS_qwen2_5", "")),
    "mistral": _split_backends(os.getenv("BACKENDS_mistral", "")),
}
VARIANTS = {k: v for k, v in VARIANTS.items() if v}

TOP_K_DEFAULT = _safe_int(os.getenv("TOP_K", "6"), 6)

GRPC_MAX_MSG = _safe_int(os.getenv("GRPC_MAX_MSG_BYTES", str(64 * 1024 * 1024)), 64 * 1024 * 1024)
CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", GRPC_MAX_MSG),
    ("grpc.max_receive_message_length", GRPC_MAX_MSG),
)
SERVER_OPTIONS = (
    ("grpc.max_send_message_length", GRPC_MAX_MSG),
    ("grpc.max_receive_message_length", GRPC_MAX_MSG),
)


class RouterServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self):
        if not VARIANTS:
            raise RuntimeError(
                "No backends configured. Set BACKENDS_llama3 / BACKENDS_qwen2_5 / BACKENDS_mistral."
            )

        self.rr = {k: itertools.cycle(v) for k, v in VARIANTS.items()}

        self.default_variant = os.getenv("DEFAULT_VARIANT", "llama3").strip()
        if self.default_variant not in self.rr:
            self.default_variant = next(iter(self.rr.keys()))

        print(f"Router initialized with variants: {list(VARIANTS.keys())}", flush=True)
        for variant, backends in VARIANTS.items():
            print(f"  • {variant}: {backends}", flush=True)
        print(f"Default variant: {self.default_variant}", flush=True)

    def _is_auto(self, v: str) -> bool:
        v = (v or "").strip().lower()
        return (v == "") or (v == "auto") or (v == "default")

    def _choose_variant(self, md: dict, request) -> tuple[str, dict]:
        requested = (md.get("variant") or "").strip()
        if requested and not self._is_auto(requested):
            return requested, {"route": "forced", "requested_variant": requested}

        dataset = (md.get("dataset") or "").strip()
        rk = _safe_int(md.get("retrieval_k"), TOP_K_DEFAULT)

        chosen, info = pick_backend(
            dataset=dataset,
            question=getattr(request, "question", ""),
            context=getattr(request, "context", ""),
            retrieval_k=rk,
            default_variant=self.default_variant,
        )
        return chosen, info

    async def Answer(self, request, context):
        md = dict(context.invocation_metadata())
        variant, route_info = self._choose_variant(md, request)

        print(json.dumps({
            "event": "router_decision",
            "dataset": md.get("dataset"),
            "requested_variant": md.get("variant"),
            "chosen_variant": variant,
            "route_info": route_info,
            "question_chars": len(getattr(request, "question", "") or ""),
            "context_chars": len(getattr(request, "context", "") or ""),
        }), flush=True)

        if variant not in self.rr:
            available = ", ".join(self.rr.keys())
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unknown variant '{variant}'. Available: {available}",
            )

        context.set_trailing_metadata((
            ("routed-variant", variant),
            ("routed-by", str(route_info.get("route", "unknown"))),
            ("router-type", str(route_info.get("router_type", ""))),
        ))

        fwd_md = list(context.invocation_metadata())
        fwd_md.append(("routed-variant", variant))

        backends = VARIANTS[variant]
        last_err = None

        for _ in range(len(backends)):
            backend = next(self.rr[variant])
            print(f"Routing [{variant}] -> {backend}", flush=True)

            try:
                async with grpc.aio.insecure_channel(backend, options=CHANNEL_OPTIONS) as channel:
                    stub = qa_pb2_grpc.QAServerStub(channel)
                    return await stub.Answer(request, metadata=fwd_md)
            except grpc.aio.AioRpcError as e:
                last_err = e
                print(f"Backend failed {backend}: {e}", flush=True)

        details = last_err.details() if last_err else "unknown error"
        context.abort(
            grpc.StatusCode.UNAVAILABLE,
            f"All backends for '{variant}' unavailable: {details}",
        )


async def serve():
    server = grpc.aio.server(options=SERVER_OPTIONS)
    qa_pb2_grpc.add_QAServerServicer_to_server(RouterServicer(), server)
    server.add_insecure_port("[::]:50050")
    await server.start()
    print("Router running on port 50050", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
