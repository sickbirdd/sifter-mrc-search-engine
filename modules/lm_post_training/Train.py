import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)

import torch
from transformers import AdamW
from tqdm import tqdm  # for our progress bar
from transformers import BertForPreTraining
from modules.lm_post_training.preprocessor import Preprocessor as pp

modelName = conf["model"]["name"]
postTrainingPreprocessor = pp(modelName)

# bert 모델 불러오기
tokenizer = postTrainingPreprocessor.tokenizer
model = BertForPreTraining.from_pretrained(modelName)

#json 데이터 추출
dataPath = conf["dataset"]["post_training"]["test"]["path"]
dataDom = conf["dataset"]["post_training"]["test"]["struct"].split('/')
postTrainingPreprocessor.readData(dataPath=dataPath, dataDOM=dataDom)
train_contexts = postTrainingPreprocessor.getRawData()

# NSP
train_contexts = postTrainingPreprocessor.nextSentencePrediction(size=1000)

# 데이터 정제
refine_datas = []
for train_context in train_contexts:
    step = dict()
    step["first"] = postTrainingPreprocessor.removeSpecialCharacters(train_context["first"])
    step["second"] = postTrainingPreprocessor.removeSpecialCharacters(train_context["second"])
    refine_datas.append(step)

# 데이터 토크나이징
token_datas = []
for refine_data in refine_datas:
    token_datas.append(postTrainingPreprocessor.masking(
        tokenizer(refine_data["first"],
                  refine_data["second"],
                  add_special_tokens=True,
                  truncation=True,
                  max_length=512,
                  padding="max_length",
                  return_tensors="pt")
        ))

for token_data, train_context in zip(token_datas, train_contexts):
    token_data['next_sentence_label'] = torch.LongTensor([[train_context['label']]])

print(token_datas[0])
print(token_datas[1])
print(token_datas[2])
print(token_datas[3])
print(token_datas[4])
# # verse 8-
loader = torch.utils.data.DataLoader(token_datas, batch_size=1, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()

# initialize optimizer
optim = AdamW(model.parameters(), lr=5e-5)
epochs = conf["developments"]["epochs"]

# post-training
for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        # process
        outputs = model(input_ids=input_ids, 
                    token_type_ids=token_type_ids, 
                    attention_mask=attention_mask, 
                    next_sentence_label=next_sentence_label, 
                    labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

#===================================

# model.save_pretrained("saving_folder")

# # verse 14
# from transformers import pipeline

# unmasker = pipeline('fill-mask', model=modelName)
# unmasker(sentence)

# # verse 15
# from transformers import TrainingArguments

# args = TrainingArguments(
#     output_dir='out',
#     per_device_train_batch_size=16,
#     num_train_epochs=2
# )

# # verse 16
# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=token_datas
# )

# # verse 17
# trainer.train()

# # verse 18
# from transformers import pipeline

# unmasker = pipeline('fill-mask', model=modelName)
# unmasker(sentence)