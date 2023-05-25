from modules.mrc_service.file_parser.parser_interface import Parser
from modules.mrc_service.file_parser.utils import singleton

from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
from io import BytesIO

@singleton
class PDFParser(Parser):
    def __init__(self) -> None:
        self._resMgr = PDFResourceManager()
        self._retData = io.StringIO()
        self._TxtConverter= TextConverter(self._resMgr, self._retData, laparams= LAParams())
        self._interpreter = PDFPageInterpreter(self._resMgr, self._TxtConverter)

    def parser_buffer(self, buffer):
        return self.pdf_to_text(buffer)
    
    def parse(self, buffer, length, cond_split="\n\n") -> list:
        content = self.parser_buffer(buffer)
        paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
        return paragragh_list
    
    def pdf_to_text(self, buffer):
        file = BytesIO(buffer)
        for page in PDFPage.get_pages(file):
            self._interpreter.process_page(page)
    
        txt = self._retData.getvalue()
        return txt
        