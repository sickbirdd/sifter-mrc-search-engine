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
tokenizer = fine_tuning_module.tokenizer
mrc_datasets = load_dataset(conf['data_path'])
model = AutoModelForQuestionAnswering.from_pretrained(conf['train_model_name'])

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