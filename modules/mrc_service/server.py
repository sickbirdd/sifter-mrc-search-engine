
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.requests import Request
from search_api import title_and_context
from transformers import pipeline
import asyncio

from file_parser.parser_manager import ParserManager
from file_parser.pdf_parser import PDFParser
from file_parser.docx_parser import DocxParser
from file_parser.hwp_parser import HwpParser

MODEL_NAME = "Kdogs/klue-finetuned-squad_kor_v1"
MAX_TOP_K = 10
MAX_DOC_PAGE_SIZE = 10
DOMAINS = ["Sports, IT, ERICA"] #TODO ENUM
ALLOWED_EXTENSIONS = set(['pdf', 'docx', 'hwp']) # 허용된 확장자 관리

app = Starlette()

# localhost:8080/inference?question="..."&context="..." => queue에 질문, 문장 등록
@app.route("/inference", methods=['GET'])
async def inference(request: Request):
    """ 추론 작업

        example) localhost:8080/inference?question="..."&context="..."
    """

    # GET parameter 검증
    parameters = request.query_params
    question = parameters.get('question')
    if question == None:
        raise HTTPException(status_code=400, detail="Question은 필수정보 입니다.")

    top_k = parameters.get('top_k')
    top_k = MAX_TOP_K if top_k == None else int(top_k)
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail="top_k 속성은 [1,{}]만 허용합니다.".format(MAX_TOP_K))
    
    doc_page_size = parameters.get('doc_page_size')
    doc_page_size = MAX_DOC_PAGE_SIZE if doc_page_size == None else int(doc_page_size)
    if doc_page_size < 1 or doc_page_size > MAX_DOC_PAGE_SIZE:
        raise HTTPException(status_code=400, detail="doc_page_size 속성은 [1,{}]만 허용합니다.".format(MAX_DOC_PAGE_SIZE))
    
    # TODO 
    # domain = parameters.get('domain')
    
    try:
        documents = title_and_context(question, doc_page_size)
    except:
        raise HTTPException(status_code=404, detail="검색된 문서가 없습니다.")

    # 모델에 요청 보내기
    response_q = asyncio.Queue()
    await request.app.model_queue.put((response_q, [question for _ in range(len(documents["content"]))], documents["content"], top_k))

    # 예측 결과값 수령
    outputs = await response_q.get()
    if(len(documents["content"]) == 1):
        outputs = [outputs]
    output = []
    for article_idx, article in enumerate(outputs):
        for answer in article:
            answer["index"] = article_idx
            output.append(answer)
    output = sorted(output, key=lambda data:data.get('score'), reverse=True)
    for answer in output[:top_k]:
        answer["title"] = documents["title"][answer["index"]]
        answer["content"] = documents["content"][answer["index"]]
    return JSONResponse(output[:top_k])

@app.route("/inference", methods=['POST'])
async def inference(request: Request):
    body = await request.json()
    if not "question" in body or not "context" in body:
        raise HTTPException(status_code=400, detail="Question과 Context 필수정보 입니다.")
    
    question = body.get("question")
    context = body.get("context")
    
    top_k = body.get('top_k')
    top_k = MAX_TOP_K if top_k == None else int(top_k)
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail="top_k 속성은 [1,{}]만 허용합니다.".format(MAX_TOP_K))
    
    # 모델에 요청 보내기
    response_q = asyncio.Queue()
    await request.app.model_queue.put((response_q, question, context, top_k))

    # 예측 결과값 수령
    output = await response_q.get()

    return JSONResponse(output)

# POST localhost:8080/inference/file + form data("question": ..., "file": [file])
# TODO pdf_parser => parser_factory
@app.route("/inference/file", methods=['POST'])
async def inference_attach_file(request):
    """파일에서 질문 MRC 진행"""

    async with request.form(max_files=1000, max_fields=1000) as form:
        if not "question" in form:
            raise HTTPException(status_code=400, detail="Question은 필수정보 입니다.")
        question = form.get('question')

        top_k = form.get('top_k')
        top_k = MAX_TOP_K if top_k == None else int(top_k)
        if top_k < 1 or top_k > MAX_TOP_K:
            raise HTTPException(status_code=400, detail="top_k 속성은 [1,{}]만 허용합니다.".format(MAX_TOP_K))

        if not "file" in form:
            raise HTTPException(status_code=400, detail="파일... 주세요...")
        
        contents = await form["file"].read()

        format = form['file'].filename.split('.')[-1]
        try:
            if format == 'pdf':
                content = ParserManager(Parser=PDFParser()).execute(contents)
            elif format == 'docx':
                content = ParserManager(Parser=DocxParser()).execute(contents)
            elif format == 'hwp':
                content = ParserManager(Parser=HwpParser()).execute(contents)
            else:
                raise HTTPException(status_code=400, detail="허용되지 않은 확장자")
        except:
            raise HTTPException(status_code=400, detail="이상한 파일")

        # 모델에 요청 보내기
        response_q = asyncio.Queue()
        await request.app.model_queue.put((response_q, [question for _ in range(len(content))], content, top_k))

        # 예측 결과값 수령
        outputs = await response_q.get()
        if(len(content) == 1):
            outputs = [outputs]
        output = []
        for result in outputs:
            output.extend(result)
        output = sorted(output, key=lambda data:data.get('score'), reverse=True)
    return JSONResponse(output[:top_k])

async def server_loop(q):
    """ 서버 모델 파이프라인

    서버 모델은 병렬 처리가 되지 않는다. 즉 요청들은 기다리며 차례로 결과값을 수령해야 한다.
    
    간단히 무한히 돌면서 결과값을 반환한다.
    """
    pipe = pipeline("question-answering", model=MODEL_NAME, top_k = MAX_TOP_K)
    while True:
        (response_q, question, context, top_k) = await q.get()
        output = pipe(question=question, context=context)[:top_k]
        # print(output)
        await response_q.put(output)

@app.on_event('startup')
async def startup_event():
    """시작시 실행: 모델 로딩"""
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))