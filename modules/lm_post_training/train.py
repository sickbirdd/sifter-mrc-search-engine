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
from modules.lm_post_training.preprocessor import MeditationsDataset
from modules.lm_post_training.preprocessor import Preprocessor

model_name = conf["model"]["name"]
post_training_preprocessor = Preprocessor(model_name)

# bert 모델 불러오기
tokenizer = post_training_preprocessor.tokenizer
model = BertForPreTraining.from_pretrained(model_name)

#json 데이터 추출
data_path = conf["dataset"]["post_training"]["training"]["path"]
data_DOM = conf["dataset"]["post_training"]["training"]["struct"].split('/')
post_training_preprocessor.read_data(data_path=data_path, data_DOM=data_DOM)
train_contexts = post_training_preprocessor.get_raw_data()

# NSP
train_contexts = post_training_preprocessor.next_sentence_prediction(5000)
# size=post_training_preprocessor.get_context_size()
# size=sys.argv[1]

# 데이터 정제
refine_datas = [[], [], []]
for train_context in train_contexts:
    refine_datas[0].append(post_training_preprocessor.remove_special_characters(train_context["first"]))
    refine_datas[1].append(post_training_preprocessor.remove_special_characters(train_context["second"]))
    refine_datas[2].append(train_context["label"])

# 데이터 토크나이징 & 마스킹
token_datas = post_training_preprocessor.masking(tokenizer(refine_datas[0],
                                                           refine_datas[1],
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=512,
                                                           padding="max_length",
                                                           return_tensors="pt"
                                                           ))

token_datas["next_sentence_label"] = torch.LongTensor(refine_datas[2])

# # verse 8-
loader = torch.utils.data.DataLoader(MeditationsDataset(token_datas), batch_size=16, shuffle=True)

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