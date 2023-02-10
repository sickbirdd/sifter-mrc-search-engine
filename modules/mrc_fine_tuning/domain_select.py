import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from pathlib import Path
from datasets import load_dataset
from modules.loader import conf_ft as CONF

def parse_data(path, code):
    path = Path(path)
    
    with open(path, 'rb') as f:
        in_dict = json.load(f)
    
    target_dataset=[]
    
    for data in in_dict['data']:
        if data["doc_class"]["code"] != code:
            continue
        paragraphs = data['paragraphs'][0]
        for qas in paragraphs["qas"]:
            info={}
            info["id"]=str(paragraphs["context_id"]) + str("-") + str(qas["question_id"])
            info["title"]=data["doc_title"]
            info["context"]=paragraphs["context"].replace("\n", "")
            info["question"]=qas["question"]
            info["answers"]={"text":[qas["answers"]["text"]], "answer_start":[qas["answers"]["answer_start"]]}
            target_dataset.append(info)
            
    return target_dataset

domain_data = parse_data(CONF['dataset']['raw_path'], "스포츠")
with open(CONF['dataset']['test_path'], 'w') as f:
    json.dump(domain_data, f)
    
test_data = load_dataset("json", data_files={"test": CONF['dataset']['test_path']})
print(test_data)