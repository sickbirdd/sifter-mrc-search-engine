# mrc-search-engine
mrc-search-engine은 검색 엔진과 함께 사용되는 기계독해(Machine Reading Comprehension) 모듈을 개발하는 프로젝트입니다.

## 개발 환경 설정 가이드

가상 환경 설정(Conda)
```
conda create --name (이름) python=3.8
```

가상 환경 활성화/비활성화(Conda)
```
conda activate (이름)
conda deactivate
```

패키지 설치(CPU)
```
pip install -r requirements.txt
```

패키지 설치(GPU)
```
pip install -r requirements_gpu.txt
```

requirements.txt 생성
```
pip list --format=freeze > requirements.txt
```

<p align="center">
  [적당히 멋진 사진]
  <!-- <a href="?"><img width="420px" src="?" alt='?'></a> -->
</p>
<p align="center">
    <em>✨ MRC Inference Api Server ✨</em>
</p>
</p>

---

**문서**: 없음

---

# Inference Api

MRC 모델의 핵심적인 사용 기능입니다.

문장과 원하는 질문을 보내면 인공지능 모델을 사용하여 문장에서 해답을 찾아서 반환해 줍니다.

## 요구사항

Python
Starlette
trainsformer

## 시작

uvicorn server:app

## Inference 예제

**GET [Server Address]/inferece**:

```shell
GET http://localhost:8000/inference?question="What is a good example of a question answering dataset?"&context="Extractive Question Answering is the task of extracting an answer from a text given a question. An example of aquestion answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script."
```

```shell
[
    {
        "score": 0.9686582088470459,
        "start": 146,
        "end": 159,
        "answer": "SQuAD dataset"
    },
    {
        "score": 0.011825370602309704,
        "start": 146,
        "end": 151,
        "answer": "SQuAD"
    },
    {
        "score": 0.00457766791805625,
        "start": 142,
        "end": 159,
        "answer": "the SQuAD dataset"
    },
    {
        "score": 0.003165960079059005,
        "start": 152,
        "end": 159,
        "answer": "dataset"
    }
]
```

parameters
---

* question(**필수**) :  질문
* context(**필수**) : 문장
* top_k : 원하는 정보 개수
* domain : 사용할 도메인 

response
---

* score : 예측 점수
* start : 예측값 시작 위치
* end : 예측값 종료 위치
* answer : 답변