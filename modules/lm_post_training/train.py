import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)

    
from modules.config.logging import single_logger, logging
from modules.config.TqdmToLogger import TqdmToLogger
single_logger().setLogger('train')
logger = single_logger().getLogger()

import torch
from transformers import AdamW
from tqdm import tqdm
from transformers import BertForPreTraining
from torch.utils.data import DataLoader
from modules.lm_post_training.preprocessor import MeditationsDataset
from modules.lm_post_training.preprocessor import Preprocessor

logger.info('================== NEW TASK ======================')


model_name = conf["model"]["name"]
config = conf['post_training']
post_training_preprocessor = Preprocessor(model_name)

# bert 모델 불러오기
tokenizer = post_training_preprocessor.tokenizer
model = BertForPreTraining.from_pretrained(model_name)

#json 데이터 추출
DATA_PATH = conf["dataset"]["post_training"]["test"]["path"]
DATA_DOM = conf["dataset"]["post_training"]["test"]["struct"].split('/')
post_training_preprocessor.read_data(data_path=DATA_PATH, data_DOM=DATA_DOM)

logger.info("추출된 기사 개수: " + str(post_training_preprocessor.get_size()))
logger.info("추출된 문장 개수: " + str(post_training_preprocessor.get_context_size()))

# NSP
TRAIN_DATA_SIZE = config["context_pair_size"]
logger.info("훈련할 데이터쌍 개수: " + str(TRAIN_DATA_SIZE))
train_contexts = post_training_preprocessor.next_sentence_prediction(TRAIN_DATA_SIZE)

# 데이터 정제
refine_datas = {"first":[], "second":[], "labels":[]}
for train_context in train_contexts:
    refine_datas['first'].append(post_training_preprocessor.remove_special_characters(train_context["first"]))
    refine_datas['second'].append(post_training_preprocessor.remove_special_characters(train_context["second"]))
    refine_datas['labels'].append(train_context["label"])
logger.info("데이터 정제 완료")

# 데이터 토크나이징 & 마스킹
token_datas = post_training_preprocessor.masking(tokenizer(refine_datas['first'],
                                                           refine_datas['second'],
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=config["max_length"],
                                                           padding="max_length",
                                                           return_tensors="pt"
                                                           ))

token_datas["next_sentence_label"] = torch.LongTensor(refine_datas['labels'])
logger.info("토크나이징 & 마스킹 완료")

# 배치 사이즈만큼 데이터 로딩 
loader = DataLoader(MeditationsDataset(token_datas), batch_size=config["batch_size"], shuffle=True)

# 모델 준비
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device) # 설정한 device로 모델 로딩
model.train() # 모델 훈련 모드

# 옵티마이저 세팅
optim = AdamW(model.parameters(), lr=5e-5)
epochs = conf["developments"]["epochs"]

# 훈련(Pre-training)
logger.info("훈련 시작")
tqdm_out = TqdmToLogger(logger,level=logging.INFO)
for epoch in range(epochs):
    loop = tqdm(loader, leave=True, file=tqdm_out) # setup loop with TQDM and dataloader
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
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