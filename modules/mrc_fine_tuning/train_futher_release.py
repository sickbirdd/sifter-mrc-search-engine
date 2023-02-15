from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering
from datasets import load_dataset
from .preprocessor import Preprocessor
from .evaluator import Evaluator
from config.logging import SingleLogger, LoggerLogCallback

class FineTuning:
    """실제 Fine Tuning 훈련을 진행한다.
    
    Attributes:
        preprocessor (`Preprocessor`): ??
        evaluation (`Evaluator`): ??
    """
    def __init__(self, CONF) -> None:
        self.CONF = CONF
        self.preprocessor = Preprocessor(conf=self.CONF)
        self.evaluation = Evaluator(conf=self.CONF)
        self.LOGGER = SingleLogger().setFileogger(logger_name='train-ft', file_name="train-ft.log", level="INFO")

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
            self.LOGGER.info(str(dataset) + " 데이터 셋이 설정되었습니다.")
            
            if dataset == 'train':
                self.__train_dataset = dataset
            else:
                self.__validation_dataset = dataset
        
        return self.__train_dataset if dataset == 'train' else self.__validation_dataset
        
        
    def __load_dataset(self):
        """데이터 셋이 없으면 저장"""
        cls = type(self)
        if not hasattr(cls, "self.__mrc_dataset"):
            self.__mrc_dataset = load_dataset(self.CONF.training_path)
            self.LOGGER.info("데이터 셋이 설정되었습니다.")


    def fine_tuning_trainer(self, mode):
        """trainer 사용

        TODO: train, eval 완전 분리
        
        주의점: mode는 train과 아닌것으로 구분된다.
        """
        self.LOGGER.info("파인 튜닝 시작")  
        self.mrc_dataset = self.__load_dataset()

        if mode == 'train':
            # 훈련 및 평가 동시 진행
            # 모델을 저장하려면 push_to_hub = True와 로그인을 위한 개인 토큰 필요
            args = TrainingArguments(
                self.CONF.upload_ft,
                evaluation_strategy="no",
                save_strategy="epoch",
                per_device_train_batch_size=self.CONF.train_batch,
                per_device_eval_batch_size=self.CONF.eval_batch,
                learning_rate=self.CONF.learning_rate,
                num_train_epochs=self.CONF.epochs,
                weight_decay=self.CONF.weight_decay,
                fp16=self.CONF.fp16,
                push_to_hub=self.CONF.push_to_hub
            )
        else:
            args = TrainingArguments(
                self.CONF.upload_ft,
                overwrite_output_dir = True,
                do_train = False,
                do_predict = True,
                per_device_eval_batch_size = self.CONF.eval_batch
            )
        args.optim="adamw_torch"
        args.logging_dir='./logs'
        args.logging_steps=500

        self.LOGGER.info("파인 튜닝 트레이너 세팅 완료 및 훈련 시작")
        trainer = Trainer(
            model = AutoModelForQuestionAnswering.from_pretrained(self.CONF.model_name), 
            args = args,
            train_dataset=self.__get_dataset('train') if mode == 'train' else None,
            eval_dataset=self.__get_dataset('validation') if mode == 'train' else None,
            tokenizer=self.preprocessor.tokenizer if mode == 'train' else None, 
            callbacks=[LoggerLogCallback()],
        )

        if mode == 'train':
            trainer.train()
        
        predictions, _, _ = trainer.predict(self.__get_dataset('validation'))
        start_logits, end_logits = predictions
        self.evaluation.compute_metrics(start_logits, end_logits, self.__get_dataset('validation'), self.mrc_dataset["validation"])
        
        self.LOGGER.info("파인 튜닝 완료")