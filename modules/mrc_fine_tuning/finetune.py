import collections
import numpy as np
from evaluate import load
from transformers import AutoTokenizer
from tqdm.auto import tqdm

class fineTuningProcess:
    def __init__(self, config) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(config['train_model_name'])
        self.__maxLength = config['max_length']
        self.__stride = config['stride']
        self.__metric = load(config['metric_type'])
        self.__nBest = config['n_best']
        self.__maxAnswerLength = config['max_answer_length']
    
    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.__maxLength,
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
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.__maxLength,
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
    
    def compute_metrics(self, start_logits, end_logits, features, examples):
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
                        if end_index < start_index or end_index - start_index + 1 > self.__maxAnswerLength:
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