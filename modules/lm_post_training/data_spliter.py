import os
from pathlib import Path
import json
import kss
import pickle

class Extractor:
    """ 파일에서 문장을 추출하는 객체입니다.
    
    Attributes:
        data (list) : 데이터를 저장하고 있습니다. (2차원 리스트; 기사-문장)
        size (int) : 기사 개수
        context_size (int) : 문장 개수
        split (bool) : 문장 분리기(kss) 사용 여부 - 문장 추출시 적용 (기본값 : False)
        directory_path (str) : 데이터 저장 위치 (디렉토리, 기본값: ext-data)
        file_name (str) : 데이터 저장 위치 (파일 이름, 기본값: extract-base.data)
        overwrite (bool) : 파일 덮어쓰기 여부 (기본값: False)

    .. note::

        extractor = Extractor()

        data_DOM = ["dataset", "*", "content", "#", "sentence"] # '*', '#' 은 리스트일 떄 사용 ('*'은 기사 단위 분리자)
        
        preprocessor.read_data("dataset/train", data_DOM)

        print(preprocessor.size) # 추출된 개수 출력

        print(preprocessor.context_size) # 추출된 문장 출력
    """
    def __init__(self) -> None:
        self._data = []
        self._size = 0
        self._context_size = 0
        self.split = False
        self.directory_path = "ext-data"
        self.file_name = "extract-base.data"
        self.overwrite = False

    @property
    def data(self):
        """ 전처리 객채에 저장된 데이터를 반환한다.
        
        Returns:
            list: 2차원 데이터 리스트 ex) [기사:[문장1, 문장2, ...], ...]
        """
        return self._data
    
    def get_context(self, size):
        """ 전처리 객체에 저장된 문장 리스트를 반환한다.
        
        Args:
            size (int): 문장 리스트 길이

        Returns:
            list: 문장 리스트
        """
        res = []
        for contexts in self._data:
            for context in contexts:
                res.append(context)
                size -= 1
                if size == 0: return res
        raise Exception("사이즈만큼 불러온 데이터가 없음")
    
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

    def _context_finder(self, context_dict_and_list, data_DOM):
            """ Dict과 List로 이루어진 데이터 구조에서 특정 값 찾기

            JSON을 포함한 Dict과 List로 이루어진 구조체에서 찾길 원하는 데이터 위치를 받아 해당 데이터를 리스트로 반환해 주는 내부
            .. warning:: 3차원 이상 구조체에서는 다른 전처리기 함수와 호환되지 않는다.
            
            Args:
                context_dict_and_list (dict): 원본 데이터 구조체
                data_DOM (list): 찾고자 하는 위치 - ex) ["root", "articles", "*", "sentences", "#", "target"]'#', '*'은 리스트를 의미한다. ('*'은 기사 단위를 분리할 떄 사용)

            Returns:
                list: 2차원 이상 값 리스트
            """
            if len(data_DOM) == 0:
                result = []
                if self.split:
                    for sentence in kss.split_sentences(context_dict_and_list):
                        result.append(sentence)
                    return len(result), result
                else:
                    return 1, [context_dict_and_list]
            
            if data_DOM[0] in ['#', '*']:
                result = []
                sum = 0
                for list_component in context_dict_and_list:
                    context_count, context_list = self._context_finder(list_component, data_DOM[1:])
                    sum += context_count
                    if context_list != None:
                        if data_DOM[0] == '#':
                            result.extend(context_list)
                        else:
                            result.append(context_list)
                return sum, result
            else:
                return self._context_finder(context_dict_and_list.get(data_DOM[0]), data_DOM[1:])


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
                    print(file)
                    # TODO: dataFormat 여러개를 받아야할 때 처리 -> dataFormat을 list로 받아 밑의 코드 여러번 수행?
                    if data_format in file:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'rb') as f:
                            in_dict = json.load(f)
                        context_size, context_list = self._context_finder(in_dict, data_DOM)
                        self._data.extend(context_list)
                        self._size = self._size + len(context_list)
                        self._context_size += context_size

    def save_data(self):
        """ 추출된 데이터를 저장합니다. """
        try:
            if not os.path.exists(self.directory_path):
                print("디렉토리가 없습니다. {self.directory_path}를 생성합니다.")
                os.makedirs(self.directory_path)
        except OSError:
            print("ERROR: 디렉토리 생성 실패")

        if not self.overwrite and os.path.exists(self.directory_path + '/' + self.file_name):
            raise Exception("이미 저장된 데이터가 존재합니다. 덮어쓰기를 설정하거나 파일 이름을 바꾸세요.")

        with open(self.directory_path + '/' + self.file_name, "wb") as f:
            dump_data = {}
            dump_data["data"] = self._data
            dump_data["size"] = self._size
            dump_data["context_size"] = self._context_size
            pickle.dump(dump_data, f)
            print("데이터가 저장되었습니다.")

    def load_data(self):
        """ 데이터를 불러와 객체에 로드합니다. """
        if os.path.exists(self.directory_path + '/' + self.file_name):
            print("저장된 데이터를 찾았습니다. 복원을 시도합니다.")
            with open(self.directory_path + '/' + self.file_name, "rb") as f:
                dump_data = pickle.load(f)
                self._data = dump_data["data"]
                self._size = dump_data["size"]
                self._context_size = dump_data["context_size"]
            print("데이터를 복원하였습니다.")
        else:
            print("데이터를 찾을 수 없습니다.")