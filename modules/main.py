"""MRC Training Modules:
MRC 모델을 훈련하기 위한 프로그램

다음과 같은 훈련 프로그램을 지원합니다.

* POST-TRAING
* FINE-TUNING
* EVALUATION MODEL
"""

import yaml
import logging
from utils.logging import SingleLogger
import os
import sys
from lm_post_training.train import Trainer
from mrc_fine_tuning.train import FineTuning
import argparse
from enum import Enum
import torch

class ModuleName(Enum):
    post_training = 1
    fine_tuning = 2
    eval = 3

def main():
    print("Main Module Execute")
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    # 입력받을 인자값 설정
    parser.add_argument('type', type=str, choices=[module.name for module in ModuleName])

    group_common = parser.add_argument_group('common')
    group_common.add_argument('--epochs', type=int, default=3, help="에폭 수(기본값: 3)")
    group_common.add_argument('--max_length', type=int, default=128, help="문장 최대 길이(기본값: 128)")
    group_common.add_argument("--model_name", type=str, default='klue/bert-base', help="모델 이름(기본값: klue/bert-base)")

    group_post_training = parser.add_argument_group('post_training')
    group_post_training.add_argument('--context_pair_size', type=int, default=1000, help="NSP 훈련 문장 쌍 개수(기본값: 1000)")
    group_post_training.add_argument('--batch_size', type=int, default=16, help="배치 크기(기본값: 16)")
    group_post_training.add_argument('--dataset_path', type=str, default="datasets/lm_post_training/training/LabeledData", help="데이터 셋 경로(기본값: datasets/lm_post_training/training/LabeledData)")
    group_post_training.add_argument('--dataset_struct', type=str, default="named_entity/#/content/#/sentence", help="데이터 셋 구조(기본값: named_entity/#/content/#/sentence)")
    group_post_training.add_argument('--upload_pt', type=str, default="modules/lm_post_training/temp_model", help="모델 저장 경로 (post-training) (기본갑: modules/lm_post_training/temp_model")
    group_post_training.add_argument('--save_pretrain_path', type=str, help="전처리 데이터셋 중간 저장 경로, 없을 시 해당 기능 사용 안함")

    group_fine_tuning = parser.add_argument_group('fine_tuning')
    group_fine_tuning.add_argument('--metric_type', type=str, default="squad", help="평가 방법(기본값: squad)")
    group_fine_tuning.add_argument('--stride', type=int, default=128, help="스트라이드(기본값: 128)")
    group_fine_tuning.add_argument('--n_best', type=int, default=20, help="?(기본값: 20)")
    
    group_fine_tuning.add_argument('--max_answer_length', type=int, default=30, help="최대 답변 길이(기본값: 30)")
    group_fine_tuning.add_argument('--train_batch', type=int, default=16, help="훈련 배치 크기(기본값: 16)")
    group_fine_tuning.add_argument('--eval_batch', type=int, default=16, help="평가 배치 크기(기본값: 16)")
    group_fine_tuning.add_argument('--learning_rate', type=int, default=0.00005, help="모델 학습률(기본값: 0.00005)")
    group_fine_tuning.add_argument('--weight_decay', type=int, default=0.01, help="?(기본값: 0.01)")
    group_fine_tuning.add_argument('--fp16', type=bool, default=False, help="?(기본값: False)")
    group_fine_tuning.add_argument('--push_to_hub', type=bool, default=False, help="Hugging Face 업로드 여부(기본값: False)")
    group_fine_tuning.add_argument('--login_token', type=str, help="Hugging Face 로그인 토큰(업로드시 필요)")
    group_fine_tuning.add_argument('--upload_ft', type=str, default="modules/mrc_fine_tuning/eval_model", help="모델 저장 경로 (fine-tuning) (업로드시 필요, 기본갑: modules/mrc_fine_tuning/eval_model)")

    group_fine_tuning.add_argument('--training_path', type=str, default="squad_kor_v1", help="파인튜닝 데이터셋(기본값: squad_kor_v1)")
    group_fine_tuning.add_argument('--raw_path', type=str, default="datasets/mrc_fine_tuning/raw/TL_span_extraction.json", help="ft-dataset-raw_path(기본값: datasets/mrc_fine_tuning/raw/TL_span_extraction.json)")
    group_fine_tuning.add_argument('--test_path', type=str, default="datasets/mrc_fine_tuning/test/sports_domain_test.json", help="ft-dataset-test_path(기본값: datasets/mrc_fine_tuning/test/sports_domain_test.json)")

    args = parser.parse_args()

    with open('config_log.yaml') as f:
        """ 설정 파일 중 log 관련 설정 파일의 정보를 불러와 설정한다."""
        logging.config.dictConfig(yaml.safe_load(f))
    SingleLogger().setLogger('train')

    if args.type == 'post_training':
        print("POST_TRAINING")
        trainer = Trainer(model_name=args.model_name, 
                device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                dataset_path=args.dataset_path,
                dataset_struct=args.dataset_struct,
                context_pair_size=args.context_pair_size,
                epochs=args.epochs,
                max_length=args.max_length,
                batch_size=args.max_length,

                preprocess_dataset_path = args.save_pretrain_path
                upload_pt=args.upload_pt
                )
        trainer.fit()
    elif args.type == 'fine_tuning':
        print("FINE_TUNING")
        FineTuning(CONF=args).fine_tuning_trainer('train')
    elif args.type == 'eval':
        print("EVAL MODE")
        FineTuning(CONF=args).fine_tuning_trainer('eval')
        
if __name__ == '__main__':
    main()