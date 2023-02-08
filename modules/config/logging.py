import logging
from logging import Logger
import logging.config
import yaml
with open('modules/config.yaml') as f:
    conf = yaml.safe_load(f)

# logging 설정 (config['log'] 참고)
config = conf["log"]
logging.config.dictConfig(config)

class single_logger:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        cls = type(self)
        if not hasattr(cls, "_init"):
            self.__logger = logging.getLogger()
            cls._init = True

    def getLogger(self) -> Logger:
        return self.__logger

    def setLogger(self, logger_name):
        self.__logger = logging.getLogger(logger_name)

test_logger = logging.getLogger('test')
# 인스턴스 진입 트래커: 인스턴스 진입 시 해당 사실을 출력한다.
def tracker(func, *args, **kwargs):
    def new_func(*args, **kwargs):  
        test_logger.debug("{}이(가) 호출되었습니다. args: {} & kwargs: {}".format(
            func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return new_func

# Test 실행 트래커: 테스트 실행 시 해당 정보를 출력한다.
# param test_name: 테스트 명칭
# msg: 추가로 표시할 정보(default "Empty")
def Test(test_name, msg = "Empty"):
    def decorator(fun):
        def wrapper(*args, **kwargs):
            test_logger.debug("Test [{}] 이(가) 실행되었습니다. msg: {}".format(test_name, msg))
            return fun(*args, **kwargs)
        return wrapper
    return decorator