import os
import sys
path_modules =  os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) ))
path_root = os.path.dirname(os.path.abspath(path_modules))
sys.path.append(path_modules)
sys.path.append(path_root)

import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)

import torch
from transformers import AdamW
from tqdm import tqdm  # for our progress bar
from transformers import AutoTokenizer, AutoModelForMaskedLM
import modules.lm_post_training.preprocessor as preprocessor
import modules.lm_post_training.dataset as dataset


# bert 모델 불러오기
modelName = conf["model"]["name"]
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForMaskedLM.from_pretrained(modelName)

#json 데이터 추출
integration = preprocessor.Integration
train_contexts, train_questions, train_answers = integration.readData(conf["dataset"]["post_training"]["training"]["path"], "스포츠")
eval_contexts, eval_questions, eval_answers = integration.readData(conf["dataset"]["post_training"]["validation"]["path"], "스포츠")

# 데이터 tokenizing
inputs = tokenizer(train_contexts, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

# bert training을 위한 masking 처리
inputs = preprocessor.maskModuel.masking(inputs, 0.15)

# # verse 8
dataset = dataset.MeditationsDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

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
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
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
#     train_dataset=dataset
# )

# # verse 17
# trainer.train()

# # verse 18
# from transformers import pipeline

# unmasker = pipeline('fill-mask', model=modelName)
# unmasker(sentence)