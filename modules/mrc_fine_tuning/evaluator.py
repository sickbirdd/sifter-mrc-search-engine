import collections
import numpy as np
from evaluate import load
from tqdm.auto import tqdm
class Evaluator:
    """ 파인튜닝 평가 객체
    모델에 대한 평가를 할 수 있음.
    
    Attributes:
        conf (dict): 설정 파일
        
    """
    def __init__(self, conf) -> None:
        self.__metric = load(conf['metric_type'])
        self.__nBest = conf['n_best']
        self.__max_answer_length = conf['max_answer_length']
    
    def compute_metrics(self, start_logits, end_logits, features, examples):
        """ 모델 평가 메트릭 함수
        F1 스코어, EM 스코어 계산
        
        Args:
            start_logits (:class:`transformers.modeling_outputs.QuestionAnsweringModelOutput.start_logits`): 모델이 예측한 답 시작지점 
            end_logits (:class:`transformers.modeling_outputs.QuestionAnsweringModelOutput.end_logits`): 모델이 예측한 답 종료지점
            features (:class:`datasets.arrow_dataset.Dataset`): 전처리가 끝난 데이터셋
            examples (:class:`datasets.arrow_dataset.Dataset`): 실제 데이터셋
        
        Returns:
            dict: F1 스코어, EM 스코어         
        """
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # 해당 예제와 연관된 모든 feature에 대해서...
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -self.__nBest - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -self.__nBest - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # 본문에 완전히 포함되지 않는 답변은 생략
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # 길이가 음수거나 maxAnswerLength를 넘는 답변은 생략
                        if end_index < start_index or end_index - start_index + 1 > self.__max_answer_length:
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return self.__metric.compute(predictions=predicted_answers, references=theoretical_answers)
    