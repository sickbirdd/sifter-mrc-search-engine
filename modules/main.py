import os
import sys
import yaml
import logging
from lm_post_training.train import train
from mrc_fine_tuning.train_futher_release import FineTuning
from config.logging import SingleLogger
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

if __name__ == '__main__':
    print("Main Module Execute")
    
    with open('config_pt.yaml') as f:
        """ 설정 파일 중 post-training 관련 설정 파일의 정보를 불러와 설정한다."""
        CONF_PT = yaml.safe_load(f)

    with open('config_ft.yaml') as f:
        """ 설정 파일 중 fine-tuning 관련 설정 파일의 정보를 불러와 설정한다."""
        CONF_FT = yaml.safe_load(f)
        
    with open('config_log.yaml') as f:
        """ 설정 파일 중 log 관련 설정 파일의 정보를 불러와 설정한다."""
        CONF_LOG = yaml.safe_load(f)

    # 설정파일에서 로거 정보를 불러와 세팅한다.
    logging.config.dictConfig(CONF_LOG)

    # 인자 체크
    if len(sys.argv) < 2:
        print("인자가 부족합니다.")
        print("다음과 같은 해결책이 있습니다.")
        print("python.exe main.py [command]")
        print("command: [post-training, fine-tuning, eval]")
        sys.exit()

    # 인자에 따라 훈련 방식 변경
    #TODO : 모델 파리미터 변경
    if sys.argv[1] == 'post-training':
        print("POST_TRAINING")
        train(CONF=CONF_PT)
    elif sys.argv[1] == 'fine-tuning':
        print("FINE_TUNING")
        FineTuning(CONF=CONF_FT).fine_tuning_trainer('train')
    elif sys.argv[1] == 'eval':
        print("EVAL MODE")
        FineTuning(CONF=CONF_FT).fine_tuning_trainer('eval')
    else:
        print("올바른 명령어가 아닙니다.")
        print("다음과 같은 해결책이 있습니다.")
        print("ex) python.exe main.py post-training")