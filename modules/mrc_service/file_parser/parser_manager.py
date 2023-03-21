from modules.mrc_service.file_parser.ab_parser import Parser
import logging

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


MIN_LENGTH = 10
class ParserManager():

    
    def __init__(self, Parser: Parser) -> None:
        self.manager = Parser

    def execute(self, buffer, length = MIN_LENGTH) -> list:
        LOGGER.debug("buffer가 입력되었습니다.")

        # DO PREPROCESS ?
        # TODO

        # parsing 진행
        
        return self.manager.parse(buffer=buffer, length = length)
