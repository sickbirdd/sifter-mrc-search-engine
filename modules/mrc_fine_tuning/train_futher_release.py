import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from modules.main import CONF_FT as CONF
from modules.mrc_fine_tuning.preprocessor import Preprocessor
from modules.mrc_fine_tuning.evaluator import Evaluator

class FineTuning:
    """실제 Fine Tuning 훈련을 진행한다.
    
    Attributes:
        preprocessor (`Preprocessor`): ??
        evaluation (`Evaluator`): ??
    """
    def __init__(self, CONF) -> None:
        self.preprocessor = Preprocessor(conf=CONF, mode=CONF['parameters']['exec'])
        self.evaluation = Evaluator(conf=CONF['parameters'])

    def __get_dataset(self, dataset):
        """데이터 셋이 이미 있는 경우 해당 데이터셋을 반환 아닌 경우 데이터셋을 정제해서 저장

        주의점: dataset은 train과 아닌 것으로 구분된다.
        """
        cls = type(self)
        if not hasattr(cls, "self.__" + str(dataset) + "_dataset"):
            dataset = self.__mrc_dataset[dataset].map(
                self.preprocessor.preprocess_training_examples,
                batched=True,
                remove_columns=self.__mrc_dataset[dataset].column_names,
            )
            
            if dataset == 'train':
                self.__train_dataset = dataset
            else:
                self.__validation_dataset = dataset
        
        return self.__train_dataset if dataset == 'train' else self.__validation_dataset
        
        
    def __load_dataset(self):
        """데이터 셋이 없으면 저장"""
        cls = type(self)
        if not hasattr(cls, "self.__mrc_dataset"):
            self.__mrc_dataset = load_dataset(CONF['dataset']['training_path'])


    def fine_tuning_trainer(self, mode):
        """trainer 사용

        TODO: train, eval 완전 분리
        
        주의점: mode는 train과 아닌것으로 구분된다.
        """
        self.mrc_dataset = self.__load_dataset()

        if mode == 'train':
            # 훈련 및 평가 동시 진행
            # 모델을 저장하려면 push_to_hub = True와 로그인을 위한 개인 토큰 필요
            args = TrainingArguments(
                CONF['model']['upload'],
                evaluation_strategy="no",
                save_strategy="epoch",
                per_device_train_batch_size=CONF['parameters']['train_batch'],
                per_device_eval_batch_size=CONF['parameters']['eval_batch'],
                learning_rate=CONF['parameters']['learning_rate'],
                num_train_epochs=CONF['parameters']['epochs'],
                weight_decay=CONF['parameters']['weight_decay'],
                fp16=CONF['parameters']['fp16'],
                push_to_hub=CONF['parameters']['push_to_hub'],
            )
        else:
            args = TrainingArguments(
                CONF['model']['upload'],
                overwrite_output_dir = True,
                do_train = False,
                do_predict = True,
                per_device_eval_batch_size = CONF['parameters']['eval_batch']
            )

        trainer = Trainer(
            model = self.preprocessor.model, 
            args = args,
            
            train_dataset=self.__get_dataset('train') if mode == 'train' else None,
            eval_dataset=self.__get_dataset('validation') if mode == 'train' else None,
            tokenizer=self.preprocessor.tokenizer if mode == 'train' else None,
        )
        if mode == 'train':
            trainer.train()
        
        predictions, _, _ = trainer.predict(self.__get_dataset('validation'))
        start_logits, end_logits = predictions
        self.evaluation.compute_metrics(start_logits, end_logits, self.__get_dataset('validation'), self.mrc_dataset["validation"])