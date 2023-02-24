import requests
import json

def search_api(question: str):
    QUERY = {
        "commonQuery": question,
        "collection": {
            "sample": {
                "id": "sample",
                "collectionId": "politics",
                "synonymExpansion": 1,
                "useSynonym": 1,
                "useLa": 1,
                "similarity": "bm25",
                "searchField": [
                    "TITLE",
                    "CONTENT"
                ],
                "documentField": [
                    # "DOCID",
                    # "Date",
                    "TITLE",
                    "CONTENT",
                    # "GUBUN",
                    # "URL",
                    # "QUERY",
                    # "TERMS",
                    # "KEYWORD",
                    # "KEYWORD_COUNT",
                    # "SentiInfo",
                    # "SentiLevel",
                    # "TOPIC",
                    # "CATEGORY"
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
    result = search_api(query)
    return result.json()["sample"]["document"][0]["fields"]["TITLE"], result.json()["sample"]["document"][0]["fields"]["CONTENT"]