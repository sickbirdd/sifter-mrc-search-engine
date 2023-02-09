import logging
import logging.config
import yaml

def setUp():
    """ 설정 파일에서 logger 설정 정보를 불러와 설정한다."""
    with open('modules/config.yaml') as f:
        conf = yaml.safe_load(f)

    # logging 설정 (config['log'] 참고)
    config = conf["log"]
    logging.config.dictConfig(config)

if __name__ == '__main__':
    setUp()

class SingleLogger:
    """ 프로세스상 유일한 logger를 공유하여 사용하기 위한 객체"""
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        cls = type(self)
        if not hasattr(cls, "_init"):
            self.__logger = logging.getLogger()
            cls._init = True

    def getLogger(self) -> logging.Logger:
        """ logger를 반환한다.

        Returns:
            :class:`logging.Logger` : 설정된 로그
        """
        return self.__logger

    def setLogger(self, logger_name) -> None:    
        """ logger를 설정한다.
        
        Args:
            logger_name (String): 사용할 로그 이름
        """
        self.__logger = logging.getLogger(logger_name)

test_logger = logging.getLogger('test')

def tracker(func, *args, **kwargs):
    """인스턴스 진입 시 해당 사실을 출력한다."""
    def new_func(*args, **kwargs):  
        test_logger.debug("{}이(가) 호출되었습니다. args: {} & kwargs: {}".format(
            func.__name__, args, kwargs))
        return func(*args, **kwargs)
    return new_func


def Test(test_name, msg = "Empty"):
    """ 테스트 실행 시 해당 정보를 출력한다.

    Args:
        test_name (String): 테스트 명칭
        
        msg (String): 추가로 표시할 정보(default "Empty")
    """
    def decorator(fun):
        def wrapper(*args, **kwargs):
            test_logger.debug("Test [{}] 이(가) 실행되었습니다. msg: {}".format(test_name, msg))
            return fun(*args, **kwargs)
        return wrapper
    
    return decorator