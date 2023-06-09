"""MRC Training Modules:
MRC 모델을 훈련하기 위한 프로그램

다음과 같은 훈련 프로그램을 지원합니다.

* POST-TRAING
* FINE-TUNING
* EVALUATION MODEL


"""

import yaml
import logging
from utils.logger import SingleLogger
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
    evaluation = 3

def main():
    print("Main Module Execute")
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    # 입력받을 인자값 설정
    parser.add_argument('type', type=str, choices=[module.name for module in ModuleName])

    group_common = parser.add_argument_group('common')
    group_post_training = parser.add_argument_group('post_training')
    group_fine_tuning = parser.add_argument_group('fine_tuning')
    group_evaluation = parser.add_argument_group('evaluation')

    # 공통 사항
    group_common.add_argument('--epochs', type=int, default=3, help="에폭 수(기본값: 3)")
    group_common.add_argument('--max_length', type=int, default=128, help="문장 최대 길이(기본값: 128)")
    group_common.add_argument("--model_name", type=str, default='klue/bert-base', help="모델 이름(기본값: klue/bert-base)")

    # ============================ #
    # POST TRAINGING 관련 파라미터
    # ============================ #

    # 하이퍼 파라미터
    group_post_training.add_argument('--train_size', type=int, default=1000, help="훈련 데이터셋 크기(NSP 문장 쌍 개수 혹은 single 문장 개수)(기본값: 1000)")
    group_post_training.add_argument('--batch_size', type=int, default=16, help="배치 크기(기본값: 16)")

    # 옵션
    group_post_training.add_argument('--do_NSP', type=int, default=1, help="전처리 과정 상 다음 문장 예측 사용 여부 (기본값: True)")
    group_post_training.add_argument('--NSP_prob', type=float, default=0.5, help="다음 문장 예측 확률 (기본값: 0.5)")
    group_post_training.add_argument('--mask_prob', type=float, default=0.15, help="마스킹 확률(문장 변경 확률 - 그중 0.8 마스킹 0.1 다른 문장 0.1 변경 X) (기본값: 0.15)")

    # 데이터 셋: 모두의 말뭉치 기사 데이터 셋으로 설정되어 있습니다.
    group_post_training.add_argument('--dataset_path', type=str, default="datasets", help="데이터 셋 경로(기본값: datasets)")
    group_post_training.add_argument('--dataset_struct', type=str, default="document/*/paragraph/#/form", help="데이터 셋 구조(기본값: document/*/paragraph/#/form)")
    group_post_training.add_argument('--save_pretrain_path', type=str, help="전처리 데이터셋 중간 저장 경로, 없을 시 해당 기능 사용 안함")
    group_post_training.add_argument('--split', type=int, default=0, help="문장 분리기 사용 여부(기본값: False)")
    group_post_training.add_argument('--extract-path', type=str, help="문장 추출 결과 저장 위치")
    group_post_training.add_argument('--overwrite', type=int, default=0, help="문장 추출 결과 덮어쓰기 여부(기본값: False)")
    group_post_training.add_argument('--condition-branch', type=str, nargs='*', help="데이터 셋 검색 분기점(예시 값: document/*)")
    group_post_training.add_argument('--condition-path', type=str, nargs='*', help="데이터 셋 검색 분기점 이후 경로(예시 값: metadata/topic)")
    group_post_training.add_argument('--condition-value', type=str, nargs='*', help="데이터 셋 검색 조건 비교 값(예시 값: 스포츠)")

    # 모델 관리
    group_post_training.add_argument('--upload_pt', type=str, default="modules/lm_post_training/temp_model", help="모델 저장 경로 (post-training) (기본갑: modules/lm_post_training/temp_model")


    # ========================= #
    # Fine Tuning 관련 파라미터
    # ========================= #

    # 하이퍼 파라미터
    group_fine_tuning.add_argument('--train_batch', type=int, default=16, help="훈련 배치 크기(기본값: 16)")
    group_fine_tuning.add_argument('--eval_batch', type=int, default=16, help="평가 배치 크기(기본값: 16)")
    group_fine_tuning.add_argument('--max_answer_length', type=int, default=30, help="최대 답변 길이(기본값: 30)")
    group_fine_tuning.add_argument('--learning_rate', type=float, default=0.00005, help="모델 학습률(기본값: 0.00005)")
    group_fine_tuning.add_argument('--weight_decay', type=int, default=0.01, help="(기본값: 0.01)")
    group_fine_tuning.add_argument('--fp16', type=int, default=0, help="(기본값: False)")

    # 모델 관리
    group_fine_tuning.add_argument('--push_to_hub', type=int, default=0, help="Hugging Face 업로드 여부(기본값: False)")
    group_fine_tuning.add_argument('--login_token', type=str, help="Hugging Face 로그인 토큰(업로드시 필요)")
    group_fine_tuning.add_argument('--upload_ft', type=str, default="modules/mrc_fine_tuning/eval_model", help="모델 저장 경로 (fine-tuning) (업로드시 필요, 기본갑: modules/mrc_fine_tuning/eval_model)")
    
    # 데이터 셋
    group_fine_tuning.add_argument('--training_path', type=str, default="squad_kor_v1", help="파인튜닝 데이터셋(기본값: squad_kor_v1)")
    group_fine_tuning.add_argument('--raw_path', type=str, default="datasets/mrc_fine_tuning/raw/TL_span_extraction.json", help="ft-dataset-raw_path(기본값: datasets/mrc_fine_tuning/raw/TL_span_extraction.json)")
    group_fine_tuning.add_argument('--test_path', type=str, default="datasets/mrc_fine_tuning/test/sports_domain_test.json", help="ft-dataset-test_path(기본값: datasets/mrc_fine_tuning/test/sports_domain_test.json)")

    # ========================= #
    # Evaluation  관련 파라미터
    # ========================= #

    group_fine_tuning.add_argument('--metric_type', type=str, default="squad", help="평가 방법(기본값: squad)")
    group_fine_tuning.add_argument('--stride', type=int, default=128, help="스트라이드(기본값: 128)")
    group_fine_tuning.add_argument('--n_best', type=int, default=20, help="(기본값: 20)")
    
    args = parser.parse_args()

    with open('config_log.yaml') as f:
        """ 설정 파일 중 log 관련 설정 파일의 정보를 불러와 설정한다."""
        logging.config.dictConfig(yaml.safe_load(f))
    SingleLogger().setLogger('train')

    if (args.condition_branch != None and args.condition_path != None and args.condition_value != None):
        if (len(args.condition_branch) != len(args.condition_path)) or (len(args.condition_branch) != len(args.condition_value)):
            print("조건 문의 길이를 맞춰주세요.")
            return

    if args.type == 'post_training':
        print("POST_TRAINING")
        print(args.do_NSP)

        conditionDict = None
        if args.condition_branch != None:
            conditionDict = []
            for i in range(len(args.condition_branch)):
                conditionDict.append({"branch": args.condition_branch[i].split('/'), "path": args.condition_path[i].split('/'), "value": args.condition_value[i]})

        trainer = Trainer(model_name=args.model_name, 
                device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
                dataset_path=args.dataset_path,
                dataset_struct=args.dataset_struct.split('/'),
                train_size=args.train_size,
                epochs=args.epochs,
                max_length=args.max_length,
                batch_size=args.batch_size,

                preprocess_dataset_path = args.save_pretrain_path,
                upload_pt=args.upload_pt,

                split= True if args.split == 1 else False,
                extract_path=args.extract_path,
                overwrite= True if args.overwrite == 1 else False,

                condition=conditionDict,

                do_NSP= True if args.do_NSP == 1 else False,
                NSP_prob=args.NSP_prob,
                mask_prob=args.mask_prob
                )
        trainer.fit()
    elif args.type == 'fine_tuning':
        print("FINE_TUNING")
        FineTuning(CONF=args).fine_tuning_trainer('train')
    elif args.type == 'eval':
        print("EVALUATION")
        FineTuning(CONF=args).fine_tuning_trainer('eval')
        
if __name__ == '__main__':
    main()