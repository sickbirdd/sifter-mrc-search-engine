#Temp
import os
import re
import sys
import yaml
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

class PostTrainingPreprocessing:
    
    def __init__(self, modelName) -> None:
        self.modelName = modelName
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.__data = []
        self.__size = 0
    
    #데이터만 모두 초기화한다.
    def clear(self):
        self.__data = []
        self.__size = 0

    def getRawData(self):
        return self.__data

    #분류된 연관 문장을 SEP token과 함께 합쳐 반환한다.
    def getTokenData(self, sepToken = '[SEP]'):
        tokenData = list()
        for paragraph in self.__data:
            para = ''
            if type(paragraph) == list:
                for context in paragraph:
                    if para != '':
                        para = para + ' ' + sepToken + ' '
                    para += context
                tokenData.append(para)
            else:
                tokenData.append(paragraph)
        return tokenData

    def getSize(self):
        return self.__size
    
    # 기사 하나씩 append   
    def __contextFinder(self, contextDictAndList, dataDOM):
        # TODO: data가 그냥 원문일 시 처리 -> type을 인자로 추가, # 하나 일때 result = [[]]에 추가?
        if len(dataDOM) == 0:
            return contextDictAndList
        
        if dataDOM[0] == '#':
            result = []
            for listComponent in contextDictAndList:
                result.append(self.__contextFinder(listComponent, dataDOM[1:]))
            return result
        else:
            return self.__contextFinder(contextDictAndList.get(dataDOM[0]), dataDOM[1:])
            
    def readData(self, dataPath, dataDOM, dataFormat = ".json"):
        dataPath = Path(dataPath)
        
        for (root, directories, files) in os.walk(dataPath):
            for file in files:
                # TODO: dataFormat 여러개를 받아야할 때 처리 -> dataFormat을 list로 받아 밑의 코드 여러번 수행?
                if dataFormat in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        in_dict = json.load(f)
                    contextList = self.__contextFinder(in_dict, dataDOM)
                    self.__data.append(contextList)
                    self.__size = self.__size + len(contextList)
        
    # 불러온 데이터 정제에 사용되는 함수들
    # 함수명 변경 가능
    def removeSpecialCharacters(self):
        # 데이터를 정제할 수 있는 함수
        pass
    def nextSentencePrediction(self):
        # 사전 학습을 통해 Bert 성능을 향상시키기 위한 다음 문장 예측 기능을 수행하는 모듈을 개발한다.
        pass
    def maskedLanguageModel(self):
        # 토크나이징 한 기사 본문을 특정 비율만큼 토크나이징.
        pass 

# dataDOM = ['SJML', 'text', '#', 'content']
# labelDOM = ['named_entity', '#', 'content', '#', 'sentence']
# labelPath = "datasets/lm_post_training/training/LabeledData"

# test = PostTrainingPreprocessing('klue/bert-base')
# datas = test.readData(dataPath = labelPath, dataDOM = labelDOM)

# print("done")