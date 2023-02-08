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
from modules.config.logging import tracker, single_logger,logging

logger = single_logger().getLogger()

#NSP 관련 옵션 값을 가진 객체
class NSPMode:
    def __init__(self) -> None:
        # self.max_direct_range = 1
        self.prob = 0.5
        # no_dupplicatee: 모든 결과에서 유일한 문장 사용
        # only_first: 첫번째 문장만 유일한 문장 사용
        #TODO soft: 유일한 문장쌍 사용
        self.__strage_list = set(["no_dupplicate", "only_first", "soft"])
        self.__strategy = "no_dupplicate"

    def get_strategy_list(self):
        return self.__strageList

    def get_strategy(self):
        return self.__strategy

    def set_strategy(self, strategy):
        if strategy in self.__strage_list:
            self.__strategy = strategy
            return True
        else:
            return False

# NSP 관련 문장 정제 데이터를 저장하는 객체
class NSPDataset:
    def __init__(self, vector_type = "Dict") -> None:
        self.__vector_mode = ["Dict", "Set"]
        if not vector_type in self.__vector_mode:
            raise Exception("허용되지 않은 vector_type입니다. : ['Dict', 'Set']")

        self.vector_type = vector_type
        self.__list = dict() if vector_type == "Dict" else set()
    
    def is_used(self, index, stn_index, index_s = "", stn_index_s = ""):
        if self.vector_type == "Dict":
            return index in self.__list and (stn_index in self.__list[index] or "*" in self.__list[index])
        elif self.vector_type == "Set":
            return [index, stn_index, index_s, stn_index_s] in self.__list

    def add_list(self, index, stn_index):
        if not index in self.__list:
            self.__list[index] = set()
        self.__list[index].add(stn_index)

    def add_set(self, index_f, stn_index_f, index_s, stn_index_s):
        self.__list.add([index_f, stn_index_f, index_s, stn_index_s])
    
    def remove_list(self, index, stn_index):
        if index in self.__list and stn_index in self.__list[index]:
            self.__list[index].remove(stn_index)
            
# 전처리 데이터 셋 처리 객체
class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)            

# 전처리 처리 객체
@tracker
class Preprocessor:
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__data = []
        self.__size = 0
        self.__context_size = 0
        self.nsp_mode = NSPMode()
    
    #데이터만 모두 초기화한다.
    def clear(self):
        self.__data = []
        self.__size = 0
        self.__context_size = 0

    def get_raw_data(self):
        return self.__data

    def get_size(self):
        return self.__size
    
    def get_context_size(self):
        return self.__context_size
    
    # 기사 하나씩 append   
    def __context_finder(self, context_dict_and_list, data_DOM, deep):
        # TODO: data가 그냥 원문일 시 처리 -> type을 인자로 추가
        if len(data_DOM) == 0:
            return 1, context_dict_and_list if deep == 0 else [context_dict_and_list]
        
        if data_DOM[0] == '#':
            result = []
            sum = 0
            for list_component in context_dict_and_list:
                context_count, context_list = self.__context_finder(list_component, data_DOM[1:], deep - 1)
                sum += context_count
                result.append(context_list)
            return sum, result
        else:
            return self.__context_finder(context_dict_and_list.get(data_DOM[0]), data_DOM[1:], deep)
            
    def read_data(self, data_path, data_DOM, data_format = ".json"):
        data_path = Path(data_path)
        
        for (root, _, files) in os.walk(data_path):
            for file in files:
                # TODO: dataFormat 여러개를 받아야할 때 처리 -> dataFormat을 list로 받아 밑의 코드 여러번 수행?
                if data_format in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        in_dict = json.load(f)
                    context_size, context_list = self.__context_finder(in_dict, data_DOM, 2)
                    self.__data.extend(context_list)
                    self.__size = self.__size + len(context_list)
                    self.__context_size += context_size
        
    # 불러온 데이터 정제에 사용되는 함수들
    # 함수명 변경 가능
    def remove_special_characters(self, sentence):
        # 문장 시작과 끝 공백 제거
        def strip_sentence(sentence):
            return sentence.strip()
        # HTML 태그 제거
        def sub_tag(sentence):
            return re.sub('<[^>]*>', '', sentence)
        # 이메일 주소 제거
        def sub_email(sentence):
            return re.sub('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', sentence)
        # URL 제거
        def sub_URL(sentence):
            return re.sub('(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', sentence)
        # 꺽쇠 및 꺽쇠 안 문자 제거
        def sub_bracket(sentence):
            return re.sub(r'\<[^>]*\>', '', sentence)
        # 자음 모음 제거
        def sub_con_vow(sentence):
            return re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', sentence)
        # 공백 여러 개 하나로 치환
        def sub_blank(sentence):
            return ' '.join(sentence.split())
        # 세 번 이상 반복되는 문자 두 개로 치환
        def sub_repeat_char(sentence):
            p = re.compile('(([a-zA-Z0-9가-힣])\\2{2,})')
            result = p.findall(sentence)
            for r, _ in result:
                sentence = sentence.replace(r, r[:2])
            return sentence
        # 특수문자 제거
        def sub_noise(sentence):
            sentence = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", '', sentence)
            return sentence
        # 전체 함수 적용
        clean_methods = [strip_sentence, sub_tag, sub_email, sub_URL, sub_bracket, sub_con_vow, sub_blank, sub_repeat_char, sub_noise]
        for method in clean_methods:
            sentence = method(sentence)
        return sentence
    
    # 무작위 문장을 반환한다.
    # param exceptionIndex: 선택 제외목록
    # param permitFInal: 각 원문중 마지막 문장 선택 여부
    def __get_random_sentence(self, except_index, size = 1):
        cnt = 0
        while cnt < 1000:
            cnt = cnt + 1

            try:
                index = random.randrange(0, self.__size)
                stn_index = random.randrange(0, len(self.__data[index]) - size + 1)
                if stn_index < 0:
                    continue
            except:
                raise Exception("분류 데이터가 형식에 맞게 정제되어 있지 않습니다.")
            
            check_used = False
            for _ in range(size):
                if except_index.is_used(index, stn_index):
                    check_used = True
                    break

            if check_used:
                continue
            
            return index, stn_index
    
        raise Exception("랜덤 문장 생성 실패: 시도 초과")

    # 사전 학습을 통해 Bert 성능을 향상시키기 위한 다음 문장 예측 기능을 수행하는 모듈
    # param size: 문장 크기
    # sepToken: 문장 구분 크기
    def next_sentence_prediction(self, size):
        result = []
        result_size = 0
        used_index = NSPDataset("Set" if self.nsp_mode.get_strategy() == "soft" else "Dict")
        count = 0
        while result_size < size:
            if count == 100000:
                logger.warning("문장 생성 실패: 원하는 크기의 문장을 추출하는데 실패하였습니다. Size = " + str(result_size))
                break
            try:
                # 첫번째 문장: 무작위로 추가
                # 2번째 문장: 확률적으로 긍정 문장(원문 다음 문장), 부정 문장 추가(다른 기사의 무작위 문장)
                rand = random.random()
                if rand < self.nsp_mode.prob:
                    index_f, stn_index_f = self.__get_random_sentence(used_index, 2)
                    index_s = index_f
                    stn_index_s = stn_index_f + 1
                    label = True
                else:
                    index_f, stn_index_f = self.__get_random_sentence(used_index)
                    used_index.add_list(index_f, "*")
                    index_s, stn_index_s = self.__get_random_sentence(used_index)
                    used_index.remove_list(index_f, "*")
                    label = False
                
                step = dict()
                sentence_f = self.__data[index_f][stn_index_f]
                sentence_s = self.__data[index_s][stn_index_s]
                step['first'] = sentence_f
                step['second'] = sentence_s
                step['label'] = label
                result.append(step)
                result_size = result_size + 1
                if self.nsp_mode.get_strategy() == "no_dupplicate":
                    used_index.add_list(index_f, stn_index_f)
                    used_index.add_list(index_s, stn_index_s)
                elif self.nsp_mode.get_strategy() == "only_first":
                    used_index.add_list(index_f, stn_index_f)
                elif self.nsp_mode.get_strategy() == "soft":
                    used_index.add_set(index_f, stn_index_f, index_s, stn_index_s)
                    
            except:
                count = count + 1
        
        logger.info("실패 시도 개수: " + str(count))
        return result

    def masking(self, data_tokenizing, ratio=0.15):
        cls_token=self.tokenizer.cls_token_id
        sep_token=self.tokenizer.sep_token_id,
        mask_token=self.tokenizer.mask_token_id,
        pad_token=self.tokenizer.pad_token_id
        if type(mask_token) is tuple:
            logger.debug("tuple로 호출되었습니다.")
            mask_token = mask_token[0]

        if type(sep_token) is tuple:
            logger.debug("tuple로 호출되었습니다.")
            sep_token = sep_token[0]

        # 마스킹 전 label 키에 id 복사
        data_tokenizing['labels'] = copy.deepcopy(data_tokenizing['input_ids'])
        # 마스킹 전 label 키에 id 복사
        # data["label"] = data["input_ids"]
        # 마스킹 . 기사 하나씩


        for context in data_tokenizing["input_ids"]:
            # 15%확률로 마스킹, 특수토큰은 항상 false
            rand = torch.rand(len(context))
            mask_arr = (rand < ratio)
            for i in range(len(context)):
                if context[i] in [cls_token, sep_token, mask_token, pad_token]:
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
                        context[i] = mask_token
        return data_tokenizing