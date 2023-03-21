import requests
import json
from konlpy.tag import Mecab
import platform

#운영체제 별로 mecab dic 설치 위치 분리
if platform.system() == 'Windows':
    mecab = Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
elif platform.system() == 'Linux':
    mecab = Mecab(dicpath=r"/usr/local/lib/mecab/dic/mecab-ko-dic")


def extract_pos(sentence):
    """문장 분리"""

    pos_list = ['NNG', 'NNP', 'VV', 'VA', 'MAG','MM','NNBC','SN', 'SL']  # 추출할 품사 태그 리스트
    words = mecab.pos(sentence)
    new_words = []
    for word, pos in words:
        if pos in pos_list:
            new_words.append(word)
    return new_words

def eliminate_final_postposition(sentence: str):
    """종결 조사 제거"""
    # 제거할 품사 태그 리스트
    pos_list = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB','JKV','JKQ','JX', 'JC',   # 조사 리스트
                'SSO', 'SSC', 'SC',                                         # 특수 기호(괄호 및 구분자)
                'VCP', 'VCN', 'EF', 'EC', 'ETN', 'ETM'                      # 긍/부정 지정사 ,어말 어미
                ]  
    words = mecab.pos(sentence)
    words = list(filter(lambda word_pos: word_pos[1] in pos_list, words))

    for word_pos in reversed(words):
        word = word_pos[0]
        if sentence[-len(word):] == word:
            sentence = sentence[:-len(word)]
        else:
            break

    return sentence


def search_api(question: str, doc_page_size):
    """검색 엔진 api 서버 호출"""

    # 일상문 단어 분리
    print(question)
    word_list = extract_pos(question)
    question = ""    
    for word in word_list:
        question += word + " "
    
    print(question)
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
    print(question)
    search_documents = search_api(question, doc_page_size).json()["sample"]["document"]
    data = {"title":[], "content":[]}
    for document in search_documents:
        data["title"].append(document["fields"]["title"])
        data["content"].append(document["fields"]["content"])
    return data