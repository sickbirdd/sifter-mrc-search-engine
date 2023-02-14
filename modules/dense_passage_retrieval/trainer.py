import torch
import yaml
from model import biEncoder
import logging

class Trainer:
    """ DPR 훈련기 """
    def __init__(self) -> None:
        pass

    def loss(self):
        """ 로스 함수 계산 """
        pass

    def fit(self):
        """ 모델 훈련"""
        pass

    def evaluate(self):
        """ 모델 평가 """
        pass

    # def save(self):
    #     """ 훈련 과정 저장 """
    #     pass

    # def load(self):
    #     """ 훈련 과정 복구 """
    #     pass

if __name__ == "__main__":
    device = torch.device("cpu")
    LOGGER = logging.getLogger()
    LOGGER.addHandler(logging.StreamHandler())
    LOGGER.setLevel("DEBUG")
    LOGGER.info("hello world")
    
    with open('modules/dense_passage_retrieval/config.yaml') as f:
        CONF = yaml.safe_load(f)

    model = biEncoder(CONF)