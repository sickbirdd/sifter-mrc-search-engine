import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
with open('config_pt.yaml') as f:
    conf_pt = yaml.safe_load(f)
    
with open('config_ft.yaml') as f:
    conf_ft = yaml.safe_load(f)
    
with open('config_log.yaml') as f:
    conf_log = yaml.safe_load(f)