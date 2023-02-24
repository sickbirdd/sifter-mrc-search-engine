class Parser():
    def parser_buffer(self, buffer):
        raise NotImplementedError()
    
    def parse(self, buffer, length, cond) -> list:
        raise NotImplementedError()