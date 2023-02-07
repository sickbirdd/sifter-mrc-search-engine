import os
import re
import json
import random
import torch
import copy
from pathlib import Path
from transformers import AutoTokenizer

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from modules.config.logging import tracker, logging

#NSP 관련 옵션 값을 가진 객체
class nsp_mode:
    def __init__(self) -> None:
        # self.maxDirectRange = 1
        self.prob = 0.5
        #NoDupplicate: 모든 결과에서 유일한 문장 사용
        #OnlyFirst: 첫번째 문장만 유일한 문장 사용
        #TODO Soft: 유일한 문장쌍 사용
        self.__strageList = set(["NoDupplicate", "OnlyFirst", "Soft"])
        self.__strategy = "NoDupplicate"

    def getStrategyList(self):
        return self.__strageList

    def getStrategy(self):
        return self.__strategy

    def setStrategy(self, strategy):
        if strategy in self.__strageList:
            self.__strategy = strategy
            return True
        else:
            return False

# NSP 관련 문장 정제 데이터를 저장하는 객체
class nsp_dataset:
    def __init__(self, vectorType = "Dict") -> None:
        self.__vectorMode = ["Dict", "Set"]
        if not vectorType in self.__vectorMode:
            raise Exception("허용되지 않은 vectorType입니다. : ['Dict', 'Set']")

        self.vectorType = vectorType
        self.__list = dict() if vectorType == "Dict" else set()
    
    def isUsed(self, index, stnIndex, index_s = "", stnIndex_s = ""):
        if self.vectorType == "Dict":
            return index in self.__list and (stnIndex in self.__list[index] or "*" in self.__list[index])
        elif self.vectorType == "Set":
            return [index, stnIndex, index_s, stnIndex_s] in self.__list

    def addList(self, index, stnIndex):
        if not index in self.__list:
            self.__list[index] = set()
        self.__list[index].add(stnIndex)

    def addSet(self, index_f, stnIndex_f, index_s, stnIndex_s):
        self.__list.add([index_f, stnIndex_f, index_s, stnIndex_s])
    
    def removeList(self, index, stnIndex):
        if index in self.__list and stnIndex in self.__list[index]:
            self.__list[index].remove(stnIndex)

# 전처리 처리 객체
@tracker
class Preprocessor:
    def __init__(self, modelName) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(modelName)
        self.__data = []
        self.__size = 0
        self.__contextSize = 0
        self.nsp_mode = nsp_mode()
    
    #데이터만 모두 초기화한다.
    def clear(self):
        self.__data = []
        self.__size = 0
        self.__contextSize = 0

    def getRawData(self):
        return self.__data

    def getSize(self):
        return self.__size
    
    def getContextSize(self):
        return self.__contextSize
    
    # 기사 하나씩 append   
    def __contextFinder(self, contextDictAndList, dataDOM, deep):
        # TODO: data가 그냥 원문일 시 처리 -> type을 인자로 추가
        if len(dataDOM) == 0:
            return 1, contextDictAndList if deep == 0 else [contextDictAndList]
        
        if dataDOM[0] == '#':
            result = []
            sum = 0
            for listComponent in contextDictAndList:
                context_count, context_list = self.__contextFinder(listComponent, dataDOM[1:], deep - 1)
                sum += context_count
                result.append(context_list)
            return sum, result
        else:
            return self.__contextFinder(contextDictAndList.get(dataDOM[0]), dataDOM[1:], deep)
            
    def readData(self, dataPath, dataDOM, dataFormat = ".json"):
        dataPath = Path(dataPath)
        
        for (root, _, files) in os.walk(dataPath):
            for file in files:
                # TODO: dataFormat 여러개를 받아야할 때 처리 -> dataFormat을 list로 받아 밑의 코드 여러번 수행?
                if dataFormat in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        in_dict = json.load(f)
                    contextSize, contextList = self.__contextFinder(in_dict, dataDOM, 2)
                    self.__data.extend(contextList)
                    self.__size = self.__size + len(contextList)
                    self.__contextSize += contextSize
        
    # 불러온 데이터 정제에 사용되는 함수들
    # 함수명 변경 가능
    def removeSpecialCharacters(self, sentence):
        # 문장 시작과 끝 공백 제거
        def stripSentence(sentence):
            return sentence.strip()
        # HTML 태그 제거
        def subTag(sentence):
            return re.sub('<[^>]*>', '', sentence)
        # 이메일 주소 제거
        def subEmail(sentence):
            return re.sub('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', sentence)
        # URL 제거
        def subURL(sentence):
            return re.sub('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', sentence)
        # 꺽쇠 및 꺽쇠 안 문자 제거
        def subBracket(sentence):
            return re.sub(r'\<[^>]*\>', '', sentence)
        # 자음 모음 제거
        def subConVow(sentence):
            return re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', sentence)
        # 공백 여러 개 하나로 치환
        def subBlank(sentence):
            return ' '.join(sentence.split())
        # 세 번 이상 반복되는 문자 두 개로 치환
        def subRepeatChar(sentence):
            p = re.compile('(([a-zA-Z0-9가-힣])\\2{2,})')
            result = p.findall(sentence)
            for r, _ in result:
                sentence = sentence.replace(r, r[:2])
            return sentence
        # 특수문자 제거
        def subNoise(sentence):
            sentence = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", '', sentence)
            return sentence
        # 전체 함수 적용
        cleanMethods = [stripSentence, subTag, subEmail, subURL, subBracket, subConVow, subBlank, subRepeatChar, subNoise]
        for method in cleanMethods:
            sentence = method(sentence)
        return sentence
    
    # 무작위 문장을 반환한다.
    # param exceptionIndex: 선택 제외목록
    # param permitFInal: 각 원문중 마지막 문장 선택 여부
    def __getRandomSentence(self, exceptIndex, size = 1):
        cnt = 0
        while cnt < 1000:
            cnt = cnt + 1

            try:
                index = random.randrange(0, self.__size)
                stnIndex = random.randrange(0, len(self.__data[index]) - size + 1)
                if stnIndex < 0:
                    continue
            except:
                raise Exception("분류 데이터가 형식에 맞게 정제되어 있지 않습니다.")
            
            checkUsed = False
            for _ in range(size):
                if exceptIndex.isUsed(index, stnIndex):
                    checkUsed = True
                    break

            if checkUsed:
                continue
            
            return index, stnIndex
    
        raise Exception("랜덤 문장 생성 실패: 시도 초과")

    # 사전 학습을 통해 Bert 성능을 향상시키기 위한 다음 문장 예측 기능을 수행하는 모듈
    # param size: 문장 크기
    # sepToken: 문장 구분 크기
    def nextSentencePrediction(self, size):
        result = []
        resultSize = 0
        usedIndex = nsp_dataset("Set" if self.nsp_mode.getStrategy() == "Soft" else "Dict")
        count = 0
        while resultSize < size:
            if count == 100000:
                logging.info("문장 생성 실패: 원하는 크기의 문장을 추출하는데 실패하였습니다. Size = " + str(resultSize))
                break
            try:
                # 첫번째 문장: 무작위로 추가
                # 2번째 문장: 확률적으로 긍정 문장(원문 다음 문장), 부정 문장 추가(다른 기사의 무작위 문장)
                rand = random.random()
                if rand < self.nsp_mode.prob:
                    index_f, stnIndex_f = self.__getRandomSentence(usedIndex, 2)
                    index_s = index_f
                    stnIndex_s = stnIndex_f + 1
                    label = True
                else:
                    index_f, stnIndex_f = self.__getRandomSentence(usedIndex)
                    usedIndex.addList(index_f, "*")
                    index_s, stnIndex_s = self.__getRandomSentence(usedIndex)
                    usedIndex.removeList(index_f, "*")
                    label = False
                
                step = dict()
                sentence_f = self.__data[index_f][stnIndex_f]
                sentence_s = self.__data[index_s][stnIndex_s]
                step['first'] = sentence_f
                step['second'] = sentence_s
                step['label'] = label
                result.append(step)
                resultSize = resultSize + 1
                if self.nsp_mode.getStrategy() == "NoDupplicate":
                    usedIndex.addList(index_f, stnIndex_f)
                    usedIndex.addList(index_s, stnIndex_s)
                elif self.nsp_mode.getStrategy() == "OnlyFirst":
                    usedIndex.addList(index_f, stnIndex_f)
                elif self.nsp_mode.getStrategy() == "Soft":
                    usedIndex.addSet(index_f, stnIndex_f, index_s, stnIndex_s)
                    
            except:
                count = count + 1
        
        logging.info("실패 시도 개수: " + str(count))
        return result

    def masking(self, data_tokenizing, ratio=0.15):
        clsToken=self.tokenizer.cls_token_id
        sepToken=self.tokenizer.sep_token_id,
        maskToken=self.tokenizer.mask_token_id,
        padToken=self.tokenizer.pad_token_id
        if type(maskToken) is tuple:
            # logging.debug("tuple로 호출되었습니다.")
            maskToken = maskToken[0]

        if type(sepToken) is tuple:
            # logging.debug("tuple로 호출되었습니다.")
            sepToken = sepToken[0]

        # 마스킹 전 label 키에 id 복사
        data_tokenizing['label'] = copy.deepcopy(data_tokenizing['input_ids'])
        # 마스킹 전 label 키에 id 복사
        # data["label"] = data["input_ids"]
        # 마스킹 . 기사 하나씩


        for context in data_tokenizing["input_ids"]:
            # 15%확률로 마스킹, 특수토큰은 항상 false
            rand = torch.rand(len(context))
            mask_arr = (rand < ratio)
            for i in range(len(context)):
                if context[i] in [clsToken, sepToken, maskToken, padToken]:
                    mask_arr[i] = False
            
            for i in range(len(context)):
                token = context[i]
                mask = mask_arr[i]
                if mask == True:
                    mask_ratio = torch.rand(1)
                    # 15%의 마스킹 될 토큰 중 10%는 다른 토큰으로 대체
                    if mask_ratio <= 0.1:
                        context[i] = int(torch.randint(low=10, high=self.tokenizer.vocab_size, size=(1,)))
                    # 10%는 선택됐지만 그대로 두기    
                    elif mask_ratio <= 0.2:
                        context[i] = token
                    # 나머지는 마스크 토큰으로 변경
                    else:
                        context[i] = maskToken
        return data_tokenizing