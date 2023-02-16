import re
import os
import json
import copy
import torch
import random
from pathlib import Path
from transformers import AutoTokenizer
from modules.utils.logging import SingleLogger

class NSPMode:
    """ 다음 문장 예측 설정

    :class:`modules.lm_post_training_Preprocessor` 에서 사전 학습을 위한 다음 문장 예측에 관한 설정을 담은 Value Object

    Args:
        prob (double): 올바른 다음 문장 설정 확률
    """
    def __init__(self) -> None:
        # self.max_direct_range = 1
        self._prob = 0.5
        # no_dupplicatee: 모든 결과에서 유일한 문장 사용
        # only_first: 첫번째 문장만 유일한 문장 사용
        #TODO soft: 유일한 문장쌍 사용
        self._strage_list = set(["no_dupplicate", "only_first", "soft"])
        self._strategy = "no_dupplicate"

    @property
    def prob(self):
        """올바른 다음 문장 선택 확률"""
        return self._prob
    
    @prob.setter
    def prob(self, value : float):
        assert value >= 0.0 and value <= 1.0
        self._prob = value

    @property
    def strategy_list(self):
        """ 현재 가능한 생성 전략 리스트를 반환한다."""
        return self._strageList

    @property
    def strategy(self):
        """ 현재 생성 전략을 반환한다."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: str):
        """ 생성 전략을 설정한다.

        Args:
            strategy (String): 새로운 생성 전략 (no_dupplicate or only_first or soft)
        """
        assert value in self._strage_list
        self._strategy = value

class NSPDataset:
    """ 다음 문장 예측 설정

    :class:`modules.lm_post_training_Preprocessor` 에서 사전 학습을 위한 다음 문장 예측중 사용한 값을 담는 데이터 구조이다.

    Attributes:
        vector_type(string): 내부 데이터 구조   
    """
    def __init__(self, vector_type = "Dict") -> None:
        self._vector_mode = ["Dict", "Set"]
        if not vector_type in self._vector_mode:
            raise Exception("허용되지 않은 vector_type입니다. : ['Dict', 'Set']")

        self.vector_type = vector_type
        self._list = dict() if vector_type == "Dict" else set()
    
    def is_used(self, index, stn_index, index_s = "", stn_index_s = ""):
        """ 사용 여부 검사

        이미 사용한 값인지 검사한다.

        Returns:
            boolean: 사용한 경우 `True` 아닐 경우 `False`
        """
        if self.vector_type == "Dict":
            return index in self._list and (stn_index in self._list[index] or "*" in self._list[index])
        elif self.vector_type == "Set":
            return [index, stn_index, index_s, stn_index_s] in self._list

    def add_list(self, index, stn_index):
        """ 사용한 값 저장
        
        Args:
            index (int): 기사 id
            stn_index (int): 문장 id
        """
        if not index in self._list:
            self._list[index] = set()
        self._list[index].add(stn_index)

    def add_set(self, index_f, stn_index_f, index_s, stn_index_s):
        """ 사용한 값 저장

        Args:
            index_f (int): 기사1 id
            stn_index_f (int): 문장1 id
            index_s (int): 기사2 id
            stn_index_s (int): 문장2 id
        """
        self._list.add([index_f, stn_index_f, index_s, stn_index_s])
    
    def remove_list(self, index, stn_index):
        """ 사용한 값 삭제
        
        Args:
            index (int): 기사 id
            stn_index (int): 문장 id
        """
        if index in self._list and stn_index in self._list[index]:
            self._list[index].remove(stn_index)      

class Preprocessor: 
    """전처리를 담당하는 객체이다.

    Bert 전처리에 필요한 Masking, Next Sentence Predict(NSP)를 포함하고, 데이터 정제(특수문자 제거, 불용어 제거 등) 기능을 수행할 수 있습니다.

    Attributes:
        tokenizer (:class:`AutoTokenizer`): 전처리시 사용할 토크나이저를 저장한다.
        nsp_mode(:class:`NSPMode`): NSP 관련 설정 정보를 보유하고 있는 Value Object

    .. note::
        modelname = 'bert-base-uncased'

        preprocessor = Preprocessor(modelname)

        data_DOM = ["dataset", "#", "content", "#", "sentence"]
        
        preprocessor.read_data("dataset/train", data_DOM)

        print(preprocessor.size) # 추출된 개수 출력

        print(preprocessor.context_size) # 추출된 문장 출력
    """
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._data = []
        self._size = 0
        self._context_size = 0
        self.nsp_mode = NSPMode()
    
    #데이터만 모두 초기화한다.
    def clear(self):
        """ 전처리 객채에 저장된 데이터 모두 삭제한다."""
        self._data = []
        self._size = 0
        self._context_size = 0

    @property
    def data(self):
        """ 전처리 객채에 저장된 데이터를 반환한다.
        
        Returns:
            list: 2차원 데이터 리스트 ex) [기사:[문장1, 문장2, ...], ...]
        """
        return self._data

    @property
    def size(self):
        """ 전처리 객채에 저장된 기사 개수를 반환한다.
        
        Returns:
            int: 기사 개수
        """
        return self._size
    
    @property
    def context_size(self):
        """ 전처리 객채에 저장된 문장 개수를 반환한다.

        Returns:
            int: 문장 개수
        """
        return self._context_size
    
    def remove_special_characters(self, sentence):
        """ 토큰화 과정 전에 데이터셋을 정제하기 위한 함수
        
        Args:
            sentence (String): 정제되지 않은 문장
            
        예제:
            다음과 같이 사용하세요:
            
            >>> remove_special_characters(" #$%!MRC @프로젝트  ")
            MRC 프로젝트
            >>> remove_special_characters("bichoi0715@naver.com 메일 제거 ")
            메일 제거
            
        Returns:
            String: 정제된 문장
        """
    
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
        
        clean_methods = [strip_sentence, sub_tag, sub_email, sub_URL, sub_bracket, sub_con_vow, sub_blank, sub_repeat_char, sub_noise]
        for method in clean_methods:
            sentence = method(sentence)
        return sentence

    def _context_finder(self, context_dict_and_list, data_DOM, deep):
        """ Dict과 List로 이루어진 데이터 구조에서 특정 값 찾기

        JSON을 포함한 Dict과 List로 이루어진 구조체에서 찾길 원하는 데이터 위치를 받아 해당 데이터를 리스트로 반환해 주는 내부
        .. warning:: 3차원 이상 구조체에서는 다른 전처리기 함수와 호환되지 않는다.
        
        Args:
            context_dict_and_list (dict): 원본 데이터 구조체
            data_DOM (list): 찾고자 하는 위치 - ex) ["root", "child1", "#", "child2", "target"] "#"은 리스트를 의미한다.
            deep (int): 단순한 1차 리스트를 내부 형식 구조에 맞게 맞추기 위한 깊이

        Returns:
            list: 2차원 이상 값 리스트
        """
        if len(data_DOM) == 0:
            sentence = self.remove_special_characters(context_dict_and_list)
            # 글자 수가 5개 이하인 문장은 제외
            if len(sentence) <= 5: 
                return 0, None
            return 1, sentence if deep == 0 else [sentence]
        
        if data_DOM[0] == '#':
            result = []
            sum = 0
            for list_component in context_dict_and_list:
                context_count, context_list = self._context_finder(list_component, data_DOM[1:], deep - 1)
                sum += context_count
                if context_list != None:
                    result.append(context_list)
            return sum, result
        else:
            return self._context_finder(context_dict_and_list.get(data_DOM[0]), data_DOM[1:], deep)
            
    def read_data(self, data_path, data_DOM, data_format = ".json"):
        """ 데이터 경로에서 특정 확장자로 구성된 데이터를 모두 읽는다

        데이터가 존재하는 경로안의 파일에서 읽기 원하는 확장자를 받아 읽은 모든 JSON을 포함한 Dict과 List로 이루어진 
        구조체와 찾길 원하는 데이터 위치를 전달해 해당 데이터를 리스트로 반환한 값과 길이를 저장한다.
        
        ..warning:: 
            JSON 이외의 파일 확장자는 호환되지 않음.
        
        Args:
            data_path (str): 데이터가 있는 폴더의 경로
            data_DOM (list): 찾고자 하는 위치 - ex) ["root", "child1", "#", "child2", "target"] "#"은 리스트를 의미한다.
            data_format (str, optional): 찾고자 하는 확장자 명. 디폴트 값은 ".json".

        """
        data_path = Path(data_path)
        
        for (root, _, files) in os.walk(data_path):
            for file in files:
                # TODO: dataFormat 여러개를 받아야할 때 처리 -> dataFormat을 list로 받아 밑의 코드 여러번 수행?
                if data_format in file:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        in_dict = json.load(f)
                    context_size, context_list = self._context_finder(in_dict, data_DOM, 2)
                    self._data.extend(context_list)
                    self._size = self._size + len(context_list)
                    self._context_size += context_size
    
    def _get_random_sentence(self, except_index, size = 1):
        """ 무작위 문장 선별: NSP sub function

        내부 데이터 구조에서 무작위 문장을 선별한다. NSPused 객체를 사용하며 이미 사용한 특정 셋에 대해서는 선별하지 않는다.

        Args:
            except_index (NSPused): 선별 배제 리스트
            size (int): 찾고자 하는 다음 문장 개수

        Returns:
            tuple(int, int): 해당 기사 index와 해당 기사의 문장 index를 반환한다.
        """
        cnt = 0
        while cnt < 1000:
            cnt = cnt + 1

            try:
                index = random.randrange(0, self._size)
                stn_index = random.randrange(0, len(self._data[index]) - size + 1)
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

    def next_sentence_prediction(self, size) -> dict:
        LOGGER = SingleLogger().getLogger()
        """ 다음 문장 예측 문장 쌍 생성

        사전 학습을 위한 다음 문장 예측에 사용할 문장 쌍과 그 관계를 가진 데이터셋을 만들어 주는 함수

        Args:
            size (int): 원하는 문장 쌍 개수
        
        Returns:
            dict: 다음 문장 예측 문장 쌍 - ex) {["first": sentence1, "second": sentence2, "label": True], ...}
        """
        result = []
        result_size = 0
        used_index = NSPDataset("Set" if self.nsp_mode.strategy == "soft" else "Dict")
        count = 0
        while result_size < size:
            if count == 100000:
                LOGGER.warning("문장 생성 실패: 원하는 크기의 문장을 추출하는데 실패하였습니다. Size = " + str(result_size))
                break
            try:
                # 첫번째 문장: 무작위로 추가
                # 2번째 문장: 확률적으로 긍정 문장(원문 다음 문장), 부정 문장 추가(다른 기사의 무작위 문장)
                rand = random.random()
                if rand < self.nsp_mode.prob:
                    index_f, stn_index_f = self._get_random_sentence(used_index, 2)
                    index_s = index_f
                    stn_index_s = stn_index_f + 1
                    label = True
                else:
                    index_f, stn_index_f = self._get_random_sentence(used_index)
                    used_index.add_list(index_f, "*")
                    index_s, stn_index_s = self._get_random_sentence(used_index)
                    used_index.remove_list(index_f, "*")
                    label = False
                
                step = dict()
                sentence_f = self._data[index_f][stn_index_f]
                sentence_s = self._data[index_s][stn_index_s]
                step['first'] = sentence_f
                step['second'] = sentence_s
                step['label'] = label
                result.append(step)
                result_size = result_size + 1
                if self.nsp_mode.strategy == "no_dupplicate":
                    used_index.add_list(index_f, stn_index_f)
                    used_index.add_list(index_s, stn_index_s)
                elif self.nsp_mode.strategy == "only_first":
                    used_index.add_list(index_f, stn_index_f)
                elif self.nsp_mode.strategy == "soft":
                    used_index.add_set(index_f, stn_index_f, index_s, stn_index_s)
                    
            except:
                count = count + 1
        
        LOGGER.info("실패 시도 개수: " + str(count))
        return result

    def masking(self, data_tokenizing, ratio=0.15):
        """ 사전학습을 위해 토큰화한 데이터를 마스킹 해주는 함수

        데이터를 특정확률(15%)로 토큰ID를 마스킹 대상으로 선정한다.
        이후 15%의 마스킹 대상 토큰을 다음과 같이 바꾼다.

        
        * 10%는 다른 단어 토큰ID
        * 10%는 바꾸지 않고 그대로 자기자신 토큰ID
        * 80%는 [MASK]토큰의 토큰ID

        Args:
            data_tokenizing (dict): tokenizer를 거친 데이터 딕셔너리
            ratio (float, optional): 마스킹 비율. 디폴트 값은 0.15.

        Returns:
            dict: 마스킹 한 데이터 딕셔너리에 기존 정답 토큰ID 값인 label을 추가해 반환
        """
        cls_token=self.tokenizer.cls_token_id
        sep_token=self.tokenizer.sep_token_id,
        mask_token=self.tokenizer.mask_token_id,
        pad_token=self.tokenizer.pad_token_id
        if type(mask_token) is tuple:
            mask_token = mask_token[0]

        if type(sep_token) is tuple:
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