
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.routing import Route
from transformers import pipeline
import asyncio

MODEL_NAME = "Kdogs/klue-finetuned-squad_kor_v1"
MAX_TOP_K = 10
DOMAINS = ["Sports, IT, ERICA"] #TODO ENUM

app = Starlette()

# localhost:8080/inference?question="..."&context="..." => queue에 질문, 문장 등록
@app.route("/inference", methods=['GET'])
async def inference(request):
    """ 추론 작업

        example) localhost:8080/inference?question="..."&context="..."
    """

    # GET parameter 검증
    parameters = request.query_params
    question = parameters.get('question')
    context = parameters.get('context')
    if question == None or context == None:
        raise HTTPException(status_code=400, detail="Question과 Context는 필수정보 입니다.")

    top_k = int(parameters.get('top_k'))
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail="top_k 속성은 [1,{}]만 허용합니다.".format(MAX_TOP_K))
    
    domain = parameters.get('domain')

    # 모델에 요청 보내기
    response_q = asyncio.Queue()
    await request.app.model_queue.put((response_q, question, context, top_k))

    # 예측 결과값 수령
    output = await response_q.get()

    return JSONResponse(output)

async def server_loop(q):
    """ 서버 모델 파이프라인

    서버 모델은 병렬 처리가 되지 않는다. 즉 요청들은 기다리며 차례로 결과값을 수령해야 한다.
    
    간단히 무한히 돌면서 결과값을 반환한다.
    """
    pipe = pipeline("question-answering", model=MODEL_NAME, top_k = MAX_TOP_K)
    while True:
        (response_q, question, context, top_k) = await q.get()
        out = pipe(question=question, context=context)[:top_k]
        await response_q.put(out)

@app.on_event('startup')
async def startup_event():
    """시작시 실행: 모델 로딩"""
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))