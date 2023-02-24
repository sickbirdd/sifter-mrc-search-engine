
from tika import parser
from file_parser.ab_parser import Parser

class PDFParser(Parser):

    def parser_buffer(self, buffer):
        return parser.from_buffer(buffer)
    
    def parse(self, buffer, length, cond_split="\n\n\n") -> list:
        data = self.parser_buffer(buffer)

        content = data['content']
        paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
        return paragragh_list