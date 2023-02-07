import logging
import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)

formatters = conf["log"]["formatters"]
handlers = conf["log"]["handlers"]

logging.basicConfig(
    format=formatters['default']['format'],
    level=handlers['root']['level'],
    datefmt='%Y-%m-%d %I:%M:%S %p',
)
logger = logging.getLogger()
test_logger = logging.getLogger('test')
file_logger = logging.getLogger('file')
# file_logger.handlers.clear()
file_logger.addHandler(logging.FileHandler('log.txt', encoding="utf-8", mode='a'))

def tracker(func, *args, **kwargs):
    def new_func(*args, **kwargs):  
        logger.debug("{}이(가) 호출되었습니다. args: {} & kwargs: {}".format(
            func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return new_func

def Test(testName, msg = "Empty"):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            file_logger.debug("Test [{}] 이(가) 실행되었습니다. msg: {}".format(
                testName, msg))
            return fun(*args, **kwargs)
        return wrapper
    return decorator