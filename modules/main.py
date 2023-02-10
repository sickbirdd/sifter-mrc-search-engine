from lm_post_training.train import train
from mrc_fine_tuning.train_futher_release import FineTuning
import sys

if __name__ == '__main__':
    print("Main Module Execute")

    # 인자 체크
    if len(sys.argv) < 2:
        print("인자가 부족합니다.")
        print("다음과 같은 해결책이 있습니다.")
        print("python.exe program.py [command]")
        print("command: [post-training, fine-tuning, eval]")
        sys.exit()

    # 인자에 따라 훈련 방식 변경
    #TODO : 모델 파리미터 변경
    if sys.argv[1] == 'post-training':
        print("POST_TRAINING")
        train()
    elif sys.argv[1] == 'fine-tuning':
        print("FINE_TUNING")
        FineTuning().fine_tuning_trainer('train')
    elif sys.argv[1] == 'eval':
        print("EVAL MODE")
        FineTuning().fine_tuning_trainer('else')
    else:
        print("올바른 명령어가 아닙니다.")
        print("다음과 같은 해결책이 있습니다.")
        print("ex) python.exe program.py post-training")