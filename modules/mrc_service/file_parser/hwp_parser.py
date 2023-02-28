import olefile

from file_parser.ab_parser import Parser

class HwpParser(Parser):

    def parser_buffer(self, buffer):
        ole= olefile.OleFileIO(buffer)
        return ole.openstream('PrvText').read().decode('utf-16')
    
    def parse(self, buffer, length, cond_split="\n") -> list:
        content = self.parser_buffer(buffer)
        print(content)
        paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
        return paragragh_list