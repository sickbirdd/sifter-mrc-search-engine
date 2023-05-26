from file_parser.parser_interface import Parser
import logging

from file_parser.pdf_parser import PDFParser
from file_parser.docx_parser import DocxParser
from file_parser.hwp_parser import HwpParser
from file_parser.ppt_parser import PPTXParser
from file_parser.text_parser import TextParser 

LOGGER = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

from file_parser.utils import StrEnum
from enum import auto
    
class FileType(StrEnum):
    TEXT = auto()
    PDF = auto()
    DOCX = auto()
    HWP = auto()
    PPTX = auto()

class ParserFactory():
    """확장자에 맞춰 Parser 인스턴스를 반환해 주는 클래스입니다."""

    def is_available_file_type(self, type: str) -> bool:
        """해당 확장자를 지원해주는지 검사합니다."""
        return type.upper() in list(FileType)
    
    def getParser(self, file_type: FileType) -> Parser:
        """해당 확장자를 지원하는 Parser 인스턴스를 반환합니다."""
        if file_type == FileType.TEXT:
            return TextParser()
        elif file_type == FileType.PDF:
            return PDFParser()
        elif file_type == FileType.DOCX:
            return DocxParser()        
        elif file_type == FileType.HWP:
            return HwpParser()
        elif file_type == FileType.PPTX:
            return PPTXParser()

MIN_LENGTH = 10

class ParserManager():
    def setup(self) -> None:
        TextParser()
        PDFParser()
        DocxParser()
        HwpParser()
        PPTXParser()

    def execute(self, buffer: str, format: str, length = MIN_LENGTH) -> list:
        LOGGER.debug("buffer가 입력되었습니다.")

        format = format.upper()
        parser_factory = ParserFactory()

        if parser_factory.is_available_file_type(format) == False:
            raise ValueError()
        parser = parser_factory.getParser(FileType[format])

        return parser.parse(buffer=buffer, length=length)
