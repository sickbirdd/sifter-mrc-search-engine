import yaml
import lm_post_training
import mrc_fine_tuning
import mrc_service

with open('config.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
    print(conf)

if __name__ == '__main__':
    print("Main Module Execute")
    # LM Post Training
    # return value : transformers
    # Domain Specific BERT, koBERT(default)

    # MRC Fine Tuning

    # MRC Service