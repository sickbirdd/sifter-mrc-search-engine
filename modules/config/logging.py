import logging
import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)
logLevel = conf["log"]["level"]
logging.basicConfig(
    format='%(levelname)s[%(module)s]: %(message)s (%(asctime)s)',
    level=logLevel,
    datefmt='%Y-%m-%d %I:%M:%S %p'
)


logger = logging.getLogger()
def tracker(func, *args, **kwargs):
    def new_func(*args, **kwargs):  
        logger.debug("{}이(가) 호출되었습니다. args: {} & kwargs: {}".format(
            func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return new_func

def Test(testName, msg = "Empty"):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            logger.debug("Test [{}] 이(가) 실행되었습니다. msg: {}".format(
                testName, msg))
            return fun(*args, **kwargs)
        return wrapper
    return decorator