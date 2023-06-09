import docx2txt
from io import BytesIO
from file_parser.parser_interface import Parser
from file_parser.utils import singleton

@singleton
class DocxParser(Parser):

    def parser_buffer(self, buffer):
        file = BytesIO(buffer)
        doc = docx2txt.process(file)
        file.close()

        return doc
    
    def parse(self, buffer, length, cond_split="\n\n\n") -> list:
        content = self.parser_buffer(buffer)

        paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
        return paragragh_list