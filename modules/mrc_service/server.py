
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio

MODEL_NAME = "Kdogs/klue-finetuned-squad_kor_v1"
TOP_K = 10

app = Starlette()

# localhost:8080/inference?question="..."&context="..." => queue에 질문, 문장 등록
@app.route("/inference", methods=['GET'])
async def inference(request):
    """ 추론 작업

        example) localhost:8080/inference?question="..."&context="..."
    """
    parameters = request.query_params
    question = parameters['question']
    context = parameters['context']

    # queue에 질문, 문장 등록
    response_q = asyncio.Queue()
    await request.app.model_queue.put((question, context, response_q))

    # 결과값 수령
    output = await response_q.get()

    return JSONResponse(output)

async def server_loop(q):
    """ 서버 모델

    무한히 돌면서 결과값을 반환한다.
    """
    pipe = pipeline("question-answering", model=MODEL_NAME, top_k = TOP_K)
    while True:
        (question, context, response_q) = await q.get()
        out = pipe(question=question, context=context)
        await response_q.put(out)

@app.on_event('startup')
async def startup_event():
    """시작시 실행"""
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))