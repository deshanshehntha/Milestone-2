import os, time, asyncio, grpc, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import qa_pb2, qa_pb2_grpc

class QAServicer(qa_pb2_grpc.QAServerServicer):
    def __init__(self, model_name):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

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
        return qa_pb2.AnswerResponse(answer=ans, confidence=conf,
                                     retrieval_ms=0.0, inference_ms=inf_ms, end_to_end_ms=e2e_ms)

async def serve():
    port = int(os.environ.get("PORT","50051"))
    model = os.environ.get("MODEL_NAME","deepset/tinyroberta-squad2")
    server = grpc.aio.server()
    qa_pb2_grpc.add_QAServerServicer_to_server(QAServicer(model), server)
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"Serving {model} on :{port}", flush=True)
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
