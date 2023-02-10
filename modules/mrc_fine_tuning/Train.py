import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from modules.loader import conf_ft as CONF
from modules.mrc_fine_tuning.preprocessor import Preprocessor
from modules.mrc_fine_tuning.evaluator import Evaluator

def setUp():
    fine_tuning_module = Preprocessor(conf=CONF, mode=CONF['parameters']['exec'])
    fine_tuning_evaluation = Evaluator(conf=CONF['parameters'])

    mrc_dataset = load_dataset(CONF['dataset']['training_path'])

    train_dataset = mrc_dataset["train"].map(
        fine_tuning_module.preprocess_training_examples,
        batched=True,
        remove_columns=mrc_dataset["train"].column_names,
    )
    validation_dataset = mrc_dataset["validation"].map(
        fine_tuning_module.preprocess_validation_examples,
        batched=True,
        remove_columns=mrc_dataset["validation"].column_names,
    )
    return fine_tuning_module, train_dataset, validation_dataset, fine_tuning_evaluation, mrc_dataset

def fine_tuning_trainer():
    fine_tuning_module, train_dataset, validation_dataset, fine_tuning_evaluation, mrc_dataset = setUp()
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
    trainer = Trainer(
        model=fine_tuning_module.model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=fine_tuning_module.tokenizer,
        )
    trainer.train()
    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    fine_tuning_evaluation.compute_metrics(start_logits, end_logits, validation_dataset, mrc_dataset["validation"]) 

def fine_tuning_evaluator():
    fine_tuning_module, _, validation_dataset, fine_tuning_evaluation, mrc_dataset = setUp()
    test_args = TrainingArguments(
        CONF['model']['upload'],
        overwrite_output_dir = True,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = CONF['parameters']['eval_batch']
    )
    trainer = Trainer(
        model = fine_tuning_module.model, 
        args = test_args
        )
    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    fine_tuning_evaluation.compute_metrics(start_logits, end_logits, validation_dataset, mrc_dataset["validation"])