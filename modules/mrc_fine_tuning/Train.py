import os
import sys
path_modules =  os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) ))
path_root = os.path.dirname(os.path.abspath(path_modules))
sys.path.append(path_modules)
sys.path.append(path_root)
import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)['fine_tuning']

import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForQuestionAnswering, AutoTokenizer
import modules.mrc_fine_tuning.finetune as ft

fine_tuning_module = ft.fineTuningProcess(conf)
model_path = conf['train_model_name']
tokenizer = fine_tuning_module.tokenizer

# 훈련인지 평가인지에 따라 다른 모델 경로 설정
if(conf['exec'] == 'eval'):
    model_path = conf['eval_model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
mrc_datasets = load_dataset(conf['data_path'])

train_dataset = mrc_datasets["train"].map(
    fine_tuning_module.preprocess_training_examples,
    batched=True,
    remove_columns=mrc_datasets["train"].column_names,
)
validation_dataset = mrc_datasets["validation"].map(
    fine_tuning_module.preprocess_validation_examples,
    batched=True,
    remove_columns=mrc_datasets["validation"].column_names,
)

# 훈련 및 평가 동시 진행
# 모델을 저장하려면 push_to_hub = True와 로그인을 위한 개인 토큰 필요
if(conf['exec'] == 'train'):
    args = TrainingArguments(
        conf['upload_path'],
        evaluation_strategy="no",
        save_strategy="epoch",
        per_device_train_batch_size=conf['train_batch'],
        per_device_eval_batch_size=conf['eval_batch'],
        learning_rate=conf['learning_rate'],
        num_train_epochs=conf['epochs'],
        weight_decay=conf['weight_decay'],
        fp16=conf['fp16'],
        push_to_hub=conf['push_to_hub'],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    fine_tuning_module.compute_metrics(start_logits, end_logits, validation_dataset, mrc_datasets["validation"])

# 평가만 진행
# 평가 코드 실행 시 폴더에 파일이 생기는데 이 파일을 안지우고 실행하면 오류 뜸
else:
    test_args = TrainingArguments(
        model_path,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = conf['eval_batch'],   
        dataloader_drop_last = False    
    )
    trainer = Trainer(
        model = model, 
        args = test_args, 
        compute_metrics = fine_tuning_module.compute_metrics
    )
    test_results = trainer.predict(validation_dataset)
    print(test_results)