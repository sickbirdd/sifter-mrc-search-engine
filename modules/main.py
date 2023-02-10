from lm_post_training.train import train
from mrc_fine_tuning.Train import fine_tuning_trainer, fine_tuning_evaluator, setUp
import sys

if __name__ == '__main__':
    print("Main Module Execute")
    # LM Post Training
    # return value : transformers
    # Domain Specific BERT, koBERT(default)
    print(sys.argv)
    if len(sys.argv) < 2:
        print("인자가 부족합니다.")
        print("다음과 같은 해결책이 있습니다.")
        print("python.exe program.py [command]")
        print("command: post-training, fine-tuning, eval")
        sys.exit()

    if sys.argv[1] == 'post-training':
        print("POST_TRAINING")
        train()
    elif sys.argv[1] == 'fine-tuning':
        print("FINE_TUNING")
        fine_tuning_module, train_dataset, validation_dataset, fine_tuning_evaluation, mrc_dataset = setUp()
        fine_tuning_trainer(fine_tuning_module, train_dataset, validation_dataset, fine_tuning_evaluation, mrc_dataset)
    elif sys.argv[1] == 'eval':
        print("EVAL MODE")
        fine_tuning_module, train_dataset, validation_dataset, fine_tuning_evaluation, mrc_dataset = setUp()
        fine_tuning_evaluator(fine_tuning_module, train_dataset, validation_dataset, fine_tuning_evaluation, mrc_dataset)
    else:
        print("올바른 명령어가 아닙니다.")
        print("다음과 같은 해결책이 있습니다.")
        print("ex) python.exe program.py post-training")

    
    # MRC Fine Tuning

    # MRC Service