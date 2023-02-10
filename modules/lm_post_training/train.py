import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import yaml


    
from modules.config.logging import SingleLogger, logging
from modules.config.TqdmToLogger import TqdmToLogger
import torch
from transformers import AdamW
from tqdm import tqdm
from transformers import BertForPreTraining
from torch.utils.data import DataLoader
from dataset import MeditationsDataset
from modules.lm_post_training.preprocessor import Preprocessor
from modules.loader import conf_pt as CONF


if __name__ == '__main__':
    # with open('modules/config.yaml') as f:
    #     conf = yaml.safe_load(f)
    SingleLogger().setLogger('train')
    LOGGER = SingleLogger().getLogger()
    print(LOGGER)

    LOGGER.info('================== NEW TASK ======================')


    MODEL_NAME = CONF["model"]["name"]
    post_training_preprocessor = Preprocessor(MODEL_NAME)

    # bert 모델 불러오기
    tokenizer = post_training_preprocessor.tokenizer
    model = BertForPreTraining.from_pretrained(MODEL_NAME)

    #json 데이터 추출
    DATA_PATH = CONF["dataset"]["path"]
    DATA_DOM = CONF["dataset"]["struct"].split('/')
    post_training_preprocessor.read_data(data_path=DATA_PATH, data_DOM=DATA_DOM)

    LOGGER.info("추출된 기사 개수: " + str(post_training_preprocessor.get_size()))
    LOGGER.info("추출된 문장 개수: " + str(post_training_preprocessor.get_context_size()))

    # NSP
    TRAIN_DATA_SIZE = CONF["parameters"]["context_pair_size"]
    LOGGER.info("훈련할 데이터쌍 개수: " + str(TRAIN_DATA_SIZE))
        # 훈련시 데이터 사이즈 확인 필요
    train_contexts = post_training_preprocessor.next_sentence_prediction(100)

    # 데이터 정제
    refine_datas = {"first":[], "second":[], "labels":[]}
    for train_context in train_contexts:
        refine_datas['first'].append(post_training_preprocessor.remove_special_characters(train_context["first"]))
        refine_datas['second'].append(post_training_preprocessor.remove_special_characters(train_context["second"]))
        refine_datas['labels'].append(train_context["label"])
    LOGGER.info("데이터 정제 완료")

    # 데이터 토크나이징 (토큰 -> id)
    token_data = tokenizer(refine_datas['first'],
                            refine_datas['second'],
                            add_special_tokens=True,
                            truncation=True,
                            max_length=CONF["parameters"]["max_length"],
                            padding="max_length",
                            return_tensors="pt"
                            )
    token_data["next_sentence_label"] = torch.LongTensor(refine_datas['labels'])
    LOGGER.info("토크나이징 완료")

    # 마스킹
    mask_data = post_training_preprocessor.masking(token_data)
    LOGGER.info("마스킹 완료")


    # 배치 사이즈만큼 데이터 로딩 
    loader = DataLoader(MeditationsDataset(mask_data), batch_size=CONF["parameters"]["batch_size"], shuffle=True)

    # 모델 준비
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device) # 설정한 device로 모델 로딩
    model.train() # 모델 훈련 모드

    # 옵티마이저 세팅
    optim = AdamW(model.parameters(), lr=5e-5)
    epochs = CONF["parameters"]["epochs"]

    # 훈련(Pre-training)
    LOGGER.info("훈련 시작")
    tqdm_out = TqdmToLogger(LOGGER,level=logging.INFO)
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

    LOGGER.info("훈련이 완료되었습니다.")
    model.save_pretrained(save_directory=CONF['model']['upload'])
    tokenizer.save_pretrained(CONF['model']['upload'])