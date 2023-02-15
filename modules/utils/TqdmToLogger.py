import logging
import io

class TqdmToLogger(io.StringIO):
    """TQDM 출력 Stream을 logger로 전달하는 class
    
    Attributes:
        logger (:class:`logging.Logger`): 전달하고자 하는 logger
        level (:class:`logging.LEVEL`): 전달하고자 하는 log level
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)