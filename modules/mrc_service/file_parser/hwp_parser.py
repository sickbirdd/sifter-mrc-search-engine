# Source: https://github.com/iml1111/DocEx

import olefile
import zlib
import struct

from modules.mrc_service.file_parser.ab_parser import Parser

class HwpParser(Parser):

    def parser_buffer(self, buffer):
        file= olefile.OleFileIO(buffer)
        dirs = file.listdir()

        # HWP 파일 검증
        if ["FileHeader"] not in dirs or \
        ["\x05HwpSummaryInformation"] not in dirs:
            raise Exception("Not Valid HWP.")

        # 문서 포맷 압축 여부 확인
        header = file.openstream("FileHeader")
        header_data = header.read()
        is_compressed = (header_data[36] & 1) == 1

        # Body Sections 불러오기
        nums = []
        for d in dirs:
            if d[0] == "BodyText":
                nums.append(int(d[1][len("Section"):]))
        sections = ["BodyText/Section"+str(x) for x in sorted(nums)]

        text = ""
        for section in sections:
            bodytext = file.openstream(section)
            data = bodytext.read()
            if is_compressed:
                unpacked_data = zlib.decompress(data, -15)
            else:
                unpacked_data = data
        
            # 각 Section 내 text 추출    
            section_text = ""
            i = 0
            size = len(unpacked_data)
            while i < size:
                header = struct.unpack_from("<I", unpacked_data, i)[0]
                rec_type = header & 0x3ff
                rec_len = (header >> 20) & 0xfff

                if rec_type in [67]:
                    rec_data = unpacked_data[i+4:i+4+rec_len]
                    section_text += rec_data.decode('utf-16')
                    section_text += "\n"

                i += 4 + rec_len

            text += section_text
            text += "\n"

        return text
    
    def parse(self, buffer, length, cond_split="\n\n\n") -> list:
        content = self.parser_buffer(buffer).replace("\n", "").replace("\r", "").replace("\x02捤獥\x00\x00\x00\x00\x02\x02汤捯\x00\x00\x00\x00\x02", "")
        paragragh_list = [content]
        print(paragragh_list)
        return paragragh_list