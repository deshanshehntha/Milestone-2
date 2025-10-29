import asyncio, itertools, os, grpc, time
from concurrent import futures
import qa_pb2, qa_pb2_grpc

VARIANTS = {
    "tinyroberta": os.getenv("BACKENDS_TINYROBERTA", "localhost:50051").split(","),
    "roberta_base": os.getenv("BACKENDS_ROBERTA_BASE", "localhost:50052").split(","),
    "bert_large": os.getenv("BACKENDS_BERT_LARGE", "localhost:50053").split(","),
}

class RouterServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self):
        self.rr = {k: itertools.cycle(v) for k, v in VARIANTS.items()}

    async def Answer(self, request, context):
        md = dict(context.invocation_metadata())
        variant = md.get("variant", "tinyroberta")
        backend = next(self.rr[variant])
        async with grpc.aio.insecure_channel(backend) as channel:
            stub = qa_pb2_grpc.QAServerStub(channel)
            resp = await stub.Answer(request)
            return resp

async def serve():
    server = grpc.aio.server()
    qa_pb2_grpc.add_QAServerServicer_to_server(RouterServicer(), server)
    server.add_insecure_port("[::]:50050")
    await server.start()
    print("Router running on port 50050")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
