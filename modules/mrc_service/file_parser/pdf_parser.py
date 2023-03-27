from tika import parser
from modules.mrc_service.file_parser.parser_interface import Parser
from modules.mrc_service.file_parser.utils import singleton

@singleton
class PDFParser(Parser):

    def parser_buffer(self, buffer):
        return parser.from_buffer(buffer)
    
    def parse(self, buffer, length, cond_split="\n\n\n") -> list:
        data = self.parser_buffer(buffer)
        content = data['content']
        paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
        return paragragh_list