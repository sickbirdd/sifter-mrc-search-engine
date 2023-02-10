import os
import sys
import yaml
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    
if __name__ == '__main__':
    with open('config_pt.yaml') as f:
        """ 설정 파일 중 post-training 관련 설정 파일의 정보를 불러와 설정한다."""
        conf_pt = yaml.safe_load(f)
    
    with open('config_ft.yaml') as f:
        """ 설정 파일 중 fine-tuning 관련 설정 파일의 정보를 불러와 설정한다."""
        conf_ft = yaml.safe_load(f)
        
    with open('config_log.yaml') as f:
        """ 설정 파일 중 log 관련 설정 파일의 정보를 불러와 설정한다."""
        conf_log = yaml.safe_load(f)
else:
    conf_pt = {}
    conf_ft = {}
    conf_log = {}