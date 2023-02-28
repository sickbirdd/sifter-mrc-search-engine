import logging
import logging.config
from transformers import TrainerCallback

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

    def setLogger(self, logger_name) -> logging.Logger:    
        """ logger를 설정한다.
        
        Args:
            logger_name (String): 사용할 로그 이름
        
        Returns:
            :class:`logging.Logger` : 설정된 로그
        """
        self.__logger = logging.getLogger(logger_name)
        return self.__logger

    def setFileogger(self, logger_name, file_name = "", mode = "a", encoding="utf-8", level = "INFO", format="%(levelname)s[%(module)s]:%(message)s (%(asctime)s) pid:%(process)d pn:%(processName)s tid:%(thread)d tn:%(threadName)s") -> logging.Logger:
        """ logger를 설정한다.

        추가적인 인자를 통해 미리 정의하지 않은 파일 로그를 설정할 수 있다.
        
        Args:
            logger_name (String): 사용할 로그 이름
            file_name (String): 저장할 파일 이름
            mode (String): 파일 모드 (default : a - 이어쓰기)
            encoding (String): 파일 인코딩 (default: utf-8)

        Returns:
            :class:`logging.Logger` : 설정된 로그
        """
        self.__logger = logging.getLogger(logger_name)
        self.__logger.propagate = False
        hanler = logging.FileHandler(filename=file_name, encoding=encoding, mode=mode)
        hanler.setLevel(level)
        hanler.setFormatter(logging.Formatter(format))

        
        self.__logger.addHandler(hanler)

        self.__logger.setLevel(level)
        return self.__logger

class LoggerLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        control.should_log = False
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            SingleLogger().getLogger().info(logs)

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