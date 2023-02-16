from transformers import AutoTokenizer

class Preprocessor:
    """ 파인튜닝 전처리 객체
    학습 & 평가 데이터셋에 대한 전처리 기능이 있습니다.
    
    Attributes:
        conf (dict): 설정 인자
        
        mode (string): 훈련 or 평가 선택
        
    .. note::
        model_path = 로컬 모델 파일 경로 혹은 huggingface 모델 이름
        
        tokenizer = 전처리시 사용할 토크나이저
        
        model = 학습시키거나 평가할 모델
        
        max_length = feature의 최대 길이
        
        stride = 슬라이딩 윈도우 길이
    """
    def __init__(self, conf) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(conf.model_name)
        self.__max_length = conf.max_length
        self.__stride = conf.stride
    
    def preprocess_training_examples(self, examples):
        """ 학습 데이터셋에 대한 전처리

        Args:
            examples (:class:`datasets.formatting.formatting.LazyBatch`): 학습 데이터셋 배치
           
        Returns:
            dict : 정답 레이블이 포함된 학습 데이터셋
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.__max_length,
            truncation="only_second",
            stride=self.__stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # 본문의 처음과 끝 인덱스 찾기
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # 본문 안에 정답이 완벽하게 들어가 있지 않은 경우는 (0, 0)으로 레이블 생성
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # 그 외에는 전부 (처음 인덱스, 끝 인덱스)로 정답 레이블 생성
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
                
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    def preprocess_validation_examples(self, examples):
        """ 검증 데이터셋에 대한 전처리

        Args:
            examples (:class:`datasets.formatting.formatting.LazyBatch`): 검증 데이터셋 배치
           
        Returns:
            :class:`transformers.tokenization_utils_base.BatchEncoding` : 질문과 본문 오프셋을 구분한 검증 데이터셋
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.__max_length,
            truncation="only_second",
            stride=self.__stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    