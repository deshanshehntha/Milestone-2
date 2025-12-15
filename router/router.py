import asyncio, itertools, os, grpc, time
from concurrent import futures
import qa_pb2, qa_pb2_grpc

VARIANTS = {
    "qwen-3b": os.getenv("BACKENDS_qwen-3b", "localhost:50051").split(","),
    # "llama-3.2-3b": os.getenv("BACKENDS_llama-3.2-3b", "localhost:50052").split(","),
    "qwen-1.5b": os.getenv("BACKENDS_qwen-1.5b", "localhost:50053").split(",")
}


class RouterServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self):
        self.rr = {k: itertools.cycle(v) for k, v in VARIANTS.items()}
        print(f" Router initialized with variants: {list(VARIANTS.keys())}")
        for variant, backends in VARIANTS.items():
            print(f"  • {variant}: {list(backends)}")
    
    async def Answer(self, request, context):
        md = dict(context.invocation_metadata())
        variant = md.get("variant", "qwen-0.5b") 
        
        if variant not in self.rr:
            available = ", ".join(self.rr.keys())
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"Unknown variant '{variant}'. Available: {available}"
            )
        
        backend = next(self.rr[variant])
        print(f"Routing [{variant}] to backend: {backend}")
        
        try:
            async with grpc.aio.insecure_channel(backend) as channel:
                stub = qa_pb2_grpc.QAServerStub(channel)
                resp = await stub.Answer(request)
                return resp
        except grpc.aio.AioRpcError as e:
            print(f" Error connecting to backend {backend}: {e}")
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"Backend {backend} unavailable: {e.details()}"
            )

async def serve():
    server = grpc.aio.server()
    qa_pb2_grpc.add_QAServerServicer_to_server(RouterServicer(), server)
    server.add_insecure_port("[::]:50050")
    await server.start()
    print(" Router running on port 50050")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())