import logging
import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)

log_config = conf["log"]

logging.basicConfig(
    format=log_config['default']['format'],
    level=log_config['default']['level'],
    datefmt='%Y-%m-%d %I:%M:%S %p',
)

logger = logging.getLogger()

handler = logging.FileHandler(filename='log.txt', encoding="utf-8", mode='a')
handler.setFormatter(logging.Formatter(log_config['file']['format']))
handler.setLevel(log_config['file']['level'])
logger.addHandler(handler)

def tracker(func, *args, **kwargs):
    def new_func(*args, **kwargs):  
        logger.debug("{}이(가) 호출되었습니다. args: {} & kwargs: {}".format(
            func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return new_func

def Test(testName, msg = "Empty"):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            logger.debug("Test [{}] 이(가) 실행되었습니다. msg: {}".format(testName, msg))
            return fun(*args, **kwargs)
        return wrapper
    return decorator