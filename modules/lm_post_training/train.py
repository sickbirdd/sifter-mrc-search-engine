import torch
from tqdm import tqdm
from transformers import BertForPreTraining
from torch.utils.data import DataLoader
from .dataset import MeditationsDataset
from .preprocessor import Preprocessor
from utils.TqdmToLogger import TqdmToLogger
from utils.logging import SingleLogger, logging
import pickle
import os
import time

class Trainer:
    """Post-Training 훈련 과정"""
    def __init__(self, model_name: str, device: str, dataset_path, dataset_struct, context_pair_size: int,
                     epochs: int, max_length: int, batch_size: int, preprocess_dataset_path: str, upload_pt: str) -> None:
        self.device = device
        self.model = BertForPreTraining.from_pretrained(model_name).to(device)
        self.preprocessor = Preprocessor(model_name=model_name)
        self.preprocess_dataset_path = preprocess_dataset_path
        self.dataset_path = dataset_path
        self.dataset_struct = dataset_struct
        self.context_pair_size = context_pair_size
        self.epochs = epochs
        self.max_length = max_length
        self.batch_size = batch_size

        self.upload_pt = upload_pt

    def preprocess(self):
        """전처리기를 사용하여 데이터를 전처리한다."""
        LOGGER = SingleLogger().getLogger()

        #JSON 데이터 추출
        DATA_PATH = self.dataset_path
        DATA_DOM = self.dataset_struct.split('/')
        self.preprocessor.read_data(data_path=DATA_PATH, data_DOM=DATA_DOM)

        LOGGER.info("추출된 기사 개수: " + str(self.preprocessor.size))
        LOGGER.info("추출된 문장 개수: " + str(self.preprocessor.context_size))

        # NSP
        LOGGER.info("훈련할 데이터쌍 개수: " + str(self.context_pair_size))
            # 훈련시 데이터 사이즈 확인 필요
        train_contexts = self.preprocessor.next_sentence_prediction(self.context_pair_size)
        LOGGER.info("NSP 문장 쌍 생성 완료")

        # NSP 데이터 구조 변경
        nsp_data = {"first":[], "second":[], "labels":[]}
        for train_context in train_contexts:
            nsp_data['first'].append(train_context['first'])
            nsp_data['second'].append(train_context['second'])
            nsp_data['labels'].append(train_context['label'])
        LOGGER.info("NSP 변환 완료")
        
        # 데이터 토크나이징 (토큰 -> id)
        token_data = self.preprocessor.tokenizer(nsp_data['first'],
                                                    nsp_data['second'],
                                                    add_special_tokens=True,
                                                    truncation=True,
                                                    max_length=self.max_length,
                                                    padding="max_length",
                                                    return_tensors="pt"
                                                    )
        token_data['next_sentence_label'] = torch.LongTensor(nsp_data['labels'])
        LOGGER.info("토크나이징 완료")

        # 마스킹
        mask_data = self.preprocessor.masking(token_data)
        LOGGER.info("마스킹 완료")

        return mask_data

    def get_preprocess_dataset(self):
        """저장된 전처리 데이터가 존재하면 불러오고 없는 경우 전처리 후 데이터를 저장한다."""
        LOGGER = SingleLogger().getLogger()
        if self.preprocess_dataset_path == None:
            return self.preprocess()

        if os.path.exists(self.preprocess_dataset_path):
            LOGGER.info("이미 전처리된 데이터가 존재합니다. 해당 데이터로 학습을 진행합니다.")
        
            with open(self.preprocess_dataset_path, "rb") as f:
                dump_data = pickle.load(f)
                dump_time = dump_data['time']
                dump_size = dump_data['size']
                mask_data = dump_data['data']

            LOGGER.info("저장된 전처리 데이터 복원에 성공하였습니다. 저장된 시간: [{}], 저장된 문장 크기: {}".format(time.ctime(dump_time), dump_size))
        else:
            LOGGER.info("저장된 데이터가 없습니다. 전처리 과정이 처리된 후 해당 데이터가 저장됩니다.")
            
            mask_data =  self.preprocess()
            with open(self.preprocess_dataset_path, "wb") as f:
                dump_data = {}
                dump_data['time'] = time.time()
                dump_data['data'] = mask_data
                dump_data['size'] = mask_data['input_ids'].size()
                pickle.dump(dump_data, f)

            LOGGER.info("전처리 데이터가 성공적으로 저장되었습니다.")

        return mask_data


    def fit(self):
        """모델을 훈련한다."""
        LOGGER = SingleLogger().getLogger()

        LOGGER.info("================== NEW TASK ======================")

        # 전처리된 훈련 데이터셋 준비
        mask_data = self.get_preprocess_dataset()

        # 배치 사이즈만큼 데이터 로딩 
        loader = DataLoader(MeditationsDataset(mask_data), batch_size=self.batch_size, shuffle=True)

        # 모델 준비
        self.model.train() # 모델 훈련 모드

        # 옵티마이저 세팅
        optim = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        epochs = self.epochs

        # 훈련(Pre-training)
        LOGGER.info("훈련 시작")
        tqdm_out = TqdmToLogger(LOGGER,level=logging.INFO)
        for epoch in range(epochs):
            loop = tqdm(loader, leave=True, file=tqdm_out) # setup loop with TQDM and dataloader
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                next_sentence_label = batch['next_sentence_label'].to(self.device)
                
                # process
                outputs = self.model(input_ids=input_ids, 
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

        self.model.save_pretrained(save_directory=self.upload_pt)