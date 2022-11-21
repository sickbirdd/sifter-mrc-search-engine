#Temp
import os
import sys
path_modules =  os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) ))
path_root = os.path.dirname(os.path.abspath(path_modules))
sys.path.append(path_modules)
sys.path.append(path_root)

import json
from pathlib import Path
import torch

class Integration:
    def readData(path, code):
        path = Path(path)
        with open(path, 'rb') as f:
            in_dict = json.load(f)
        
        contexts = []
        questions = []
        answers = []
        for data in in_dict['data']:
            if code != "ALL" and data["doc_class"]["code"] != code:
                continue
            for paragraphs in data['paragraphs']:
                contexts.append(paragraphs["context"])
            for qas in paragraphs["qas"]:
                questions.append(qas["question"])
                answers.append(qas["answers"]["text"])
        return contexts, questions, answers

class maskModuel:
    def masking(inputs, ratio):
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)
        # # create mask array
        mask_arr = (rand < ratio) * (inputs.input_ids != 101) * \
                (inputs.input_ids != 102) * (inputs.input_ids != 0)
        
        # 선정된 masking 토큰 위치
        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )

        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103

        return inputs


#TODO

# import json
# from pathlib import Path

# def insertDataPathJson(path, select, __splitCondition):
#     path = Path(path)
#     with open(path, 'rb') as f:
#         in_dict = json.load(f)

#     result = []

#     idict = in_dict
#     for selectComponet in select:
#         selectPath = selectComponet.split('/')
#         idict = idict[selectPath]
#     for icomp in idict:
#         result.append(icomp)

#     return result, len(result)

# class Integration:
#     __dataSponge = list()
#     __dataSpongeSize = 0
#     __splitCondition = set()

#     def __init__(self) -> None:
#         self.__dataSponge = list()
#         self.__dataSpongeSize = 0
#         self.__splitCondition = set()

#     def clear(self):
#         self.__dataSponge.clear()
#         self.__splitCondition.clear()

#     def clearData(self):
#         self.__dataSponge = []

#     def getsplitCondition(self):
#         return self.__splitCondition
    
#     def addsplitCondition(self, cond):
#         self.__splitCondition.add(cond)

#     def delsplitCondition(self, cond):
#         if cond in self.__splitCondition :
#             self.__splitCondition.remove(cond)
    
#     def insertDataPath(self, path, code, format):
#         if format == "json":
#             dataContext = insertDataPathJson(path, code, self.__splitCondition)
#             self.__dataSponge.append(dataContext[0])
#             self.__dataSpongeSize += dataContext[1]
    
#     def insertData(self, data, format):
#         if format == "json":
#             # TODO
#             pass

#     def getAllData(self, size_in, sep):
#         # TODO

#         # result = []
#         # sentencce = ""

#         # for i in range(0, size_in):
#         #     sentence += (sep if i == 0 else "") + Integration.popData()
            
#         return self.__dataSponge

#     def popData(self):
#         if self.__dataSpongeSize == 0:
#             raise Exception('데이터가 없습니다.')
        
#         try:
#             popData = self.__dataSponge[0]
#             self.__dataSponge.pop(0)
#             self.__dataSpongeSize -= 1
#         except:
#             raise Exception("예기치 못한 오류 발생")

#         return popData

#     def popDataExtra(self, size_in, size, sep):
#         if self.__dataSpongeSize < size_in * size:
#             raise Exception('데이터가 없습니다.')
        
#         result = []
#         try:
#             for _ in range(0, size_in):
#                 sentence = ""
#                 for i in range(0, size):
#                     sentence += (sep if i == 0 else "") + self.popData()
#                 result.append(sentence)

#             self.__dataSpongeSize -= size_in * size
#         except:
#             raise Exception("예기치 못한 오류 발생")
        
#         return result

#     def toString(self) -> str:
#         return "현재 sentence 개수 : " + str(self.__dataSpongeSize) + "\n" + \
#                 "sentences : " + (' '.join(self.__dataSponge[:10]) + "..." if self.__dataSpongeSize > 10 else ' '.join(self.__dataSponge)) +  "\n" \
#                 + "현재 분리자 : " + ' '.join(self.__splitCondition)
