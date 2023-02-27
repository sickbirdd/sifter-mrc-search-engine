import requests
import json
from konlpy.tag import Mecab

def extract_pos(sentence):
    """문장 분리"""
    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
    pos_list = ['NNG', 'NNP', 'VV', 'VA', 'MAG','MM','NNBC',"SN"]  # 추출할 품사 태그 리스트
    words = mecab.pos(sentence)
    new_words = []
    for word, pos in words:
        if pos in pos_list:
            new_words.append(word)
    return new_words

def search_api(question: str):
    """검색 엔진 api 서버 호출"""

    # 일상문 단어 분리
    word_list = extract_pos(question)
    question = ""
    for word in word_list:
        question += word + " "

    QUERY = {
    "commonQuery": question,
    "collection": {
        "sample": {
            "id": "sample",
            "collectionId": "sports",
            "synonymExpansion": 1,
            "useSynonym": 1,
            "useLa": 1,
            "similarity": "bm25",
            "searchField": [
                "title",
                "content"
            ],
            "documentField": [
                "DOCID",
                "title",
                "subtitle",
                "content",
                "board",
                "writer",
                "write_date",
                "url",
                "source_site"
            ],
            "paging": {
                "from": 0,
                "size": 10
            }
        }
    }
}
 
    return requests.post("http://***REMOVED***:7000/search", data=json.dumps(QUERY))

def title_and_context(query):
    try:
        result = search_api(query)
        return result.json()["sample"]["document"][0]["fields"]["title"], result.json()["sample"]["document"][0]["fields"]["content"]
    except:
        raise Exception("검색 문서가 없습니다.")