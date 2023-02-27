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

def search_api(question: str, doc_page_size):
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
                "size": doc_page_size
            }
        }
    }
}
    return requests.post("http://***REMOVED***:7000/search", data=json.dumps(QUERY))

def title_and_context(question: str, doc_page_size)->dict:
    try:
        search_documents = search_api(question, doc_page_size).json()["sample"]["document"]
        data = {"title":[], "content":[]}
        for document in search_documents:
            data["title"].append(document["fields"]["title"])
            data["content"].append(document["fields"]["content"])
        return data
    except:
        raise Exception("검색 문서가 없습니다.")