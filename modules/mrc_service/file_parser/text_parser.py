from file_parser.parser_interface import Parser
from file_parser.utils import singleton

@singleton
class TextParser(Parser):

    def parser_buffer(self, buffer):
        return buffer.decode('utf-8')
    
    def parse(self, buffer, length, cond_split="\n\n\n") -> list:
        content = self.parser_buffer(buffer)

        paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
        return paragragh_list