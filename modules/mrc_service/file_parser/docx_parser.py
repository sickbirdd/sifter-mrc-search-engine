import docx2txt
from io import BytesIO
from modules.mrc_service.file_parser.ab_parser import Parser

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