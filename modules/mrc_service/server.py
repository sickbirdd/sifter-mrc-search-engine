
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.requests import Request
from modules.mrc_service.search_api import title_and_context
from transformers import pipeline
import asyncio

from modules.mrc_service.file_parser.parser_manager import ParserManager
from modules.mrc_service.file_parser.pdf_parser import PDFParser
from modules.mrc_service.file_parser.docx_parser import DocxParser
from modules.mrc_service.file_parser.hwp_parser import HwpParser
from modules.mrc_service.file_parser.ppt_parser import PPTXParser
from modules.mrc_service.file_parser.text_parser import TextParser 

MODEL_NAME = "Kdogs/klue-finetuned-squad_kor_v1"
MAX_TOP_K = 10
MAX_DOC_PAGE_SIZE = 10
DOMAINS = ["Sports, IT, ERICA"] #TODO ENUM
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx', 'hwp', 'pptx']) # 허용된 확장자 관리

app = Starlette()
def validate_question(question: str):
    """질문 입력값을 검증한다."""
    if question == None:
        raise HTTPException(status_code=400, detail="Question은 필수정보 입니다.")
    elif len(question) == 0:
        raise HTTPException(status_code=400, detail="Question은 공백을 허용하지 않습니다.")
    
def validate_context(context: str):
    """문장 입력값을 검증한다."""
    if context == None:
        raise HTTPException(status_code=400, detail="Context는 필수정보 입니다.")
    elif len(context) == 0:
        raise HTTPException(status_code=400, detail="Context는 공백을 허용하지 않습니다.")

    
def validate_top_k(top_k: int):
    """TOP_K 입력 값을 검증한다."""
    top_k = MAX_TOP_K if top_k == None else int(top_k)
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail="top_k 속성은 [1,{}]만 허용합니다.".format(MAX_TOP_K))
    
def validate_doc_page_size(doc_page_size: int):
    """document page size 입력 값을 검증한다."""
    doc_page_size = MAX_DOC_PAGE_SIZE if doc_page_size == None else int(doc_page_size)
    if doc_page_size < 1 or doc_page_size > MAX_DOC_PAGE_SIZE:
        raise HTTPException(status_code=400, detail="doc_page_size 속성은 [1,{}]만 허용합니다.".format(MAX_DOC_PAGE_SIZE))

# localhost:8080/inference?question="..."&context="..." => queue에 질문, 문장 등록
@app.route("/inference", methods=['GET'])
async def inference(request: Request):
    """ 추론 작업

        example) localhost:8080/inference?question="..."&context="..."
    """

    # GET parameter 검증
    parameters = request.query_params
    question = parameters.get('question')
    validate_question(question)

    top_k = parameters.get('top_k')
    top_k = MAX_TOP_K if top_k == None else int(top_k)
    validate_top_k(top_k)
    
    doc_page_size = parameters.get('doc_page_size')
    doc_page_size = MAX_DOC_PAGE_SIZE if doc_page_size == None else int(doc_page_size)
    validate_doc_page_size(doc_page_size)
    
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
    outputs = await parse_loop_message(response_q=response_q)
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
    
    question = body.get("question")
    validate_question(question)

    context = body.get("context")
    validate_context(context)
    
    top_k = body.get('top_k')
    validate_top_k(top_k)
    
    # 모델에 요청 보내기
    response_q = asyncio.Queue()
    await request.app.model_queue.put((response_q, question, context, top_k))

    # 예측 결과값 수령
    outputs = await parse_loop_message(response_q=response_q)
    for output in outputs:
        output['context'] = context

    return JSONResponse(outputs)

# POST localhost:8080/inference/file + form data("question": ..., "file": [file])
# TODO pdf_parser => parser_factory
@app.route("/inference/file", methods=['POST'])
async def inference_attach_file(request):
    """파일에서 질문 MRC 진행"""

    async with request.form(max_files=1000, max_fields=1000) as form:
        question = form.get('question')
        validate_question(question)

        top_k = form.get('top_k')
        validate_top_k(top_k)

        if not "file" in form:
            raise HTTPException(status_code=400, detail="파일이 입력되지 않은 것 같습니다.")
        
        contents = await form["file"].read()

        format = form['file'].filename.split('.')[-1]
        try:
            if format == 'txt':
                content = ParserManager(Parser=TextParser()).execute(contents)
            elif format == 'pdf':
                content = ParserManager(Parser=PDFParser()).execute(contents)
            elif format == 'docx':
                content = ParserManager(Parser=DocxParser()).execute(contents)
            elif format == 'hwp':
                content = ParserManager(Parser=HwpParser()).execute(contents)
            elif format == 'pptx':
                content = ParserManager(Parser=PPTXParser()).execute(contents, 5)
            else:
                raise HTTPException(status_code=400, detail="허용되지 않은 확장자입니다.")
        except:
            raise HTTPException(status_code=400, detail="이상한 파일: 서버 관리자에게 요청하세요.")

        # 모델에 요청 보내기
        response_q = asyncio.Queue()
        await request.app.model_queue.put((response_q, [question for _ in range(len(content))], content, top_k))

        # 예측 결과값 수령
        outputs = await parse_loop_message(response_q=response_q)
        if(len(content) == 1):
            outputs = [outputs]
        output = []
        for passage_idx, document in enumerate(outputs):
            for answer in document:
                answer["index"] = passage_idx
                output.append(answer)
        output = sorted(output, key=lambda data:data.get('score'), reverse=True)

        for answer in output[:top_k]:
            answer["content"] = content[answer["index"]]
            
    return JSONResponse(output[:top_k])

async def parse_loop_message(response_q: asyncio.Queue):
    """모델 예측 결과 메세지를 받아 필요한 값으로 변환한다."""
    # 모델 예측 성공 여부 반환
    result = await response_q.get()
    if result != True:
        raise HTTPException(status_code=400, detail="모델 예측 과정에서 예상치 못한 오류가 발생하였습니다.")
    return await response_q.get()

async def server_loop(q):
    """ 서버 모델 파이프라인

    서버 모델은 병렬 처리가 되지 않는다. 즉 요청들은 기다리며 차례로 결과값을 수령해야 한다.
    
    간단히 무한히 돌면서 결과값을 반환한다.
    """
    pipe = pipeline("question-answering", model=MODEL_NAME, top_k = MAX_TOP_K)
    while True:
        (response_q, question, context, top_k) = await q.get()
        try:
            output = pipe(question=question, context=context)[:top_k]
        except:
            await response_q.put(False)
            continue
        
        await response_q.put(True)
        await response_q.put(output)

@app.on_event('startup')
async def startup_event():
    """시작시 실행: 모델 로딩"""
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))