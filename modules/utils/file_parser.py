from tika import parser

def pdf_parser(path = "MRC_with_Unanswerable_Question.pdf", length = 10, cond_split="\n\n\n"):
    data = parser.from_file(path)

    content = data['content']
    paragragh_list = [paragragh for paragragh in content.split(cond_split) if len(paragragh) > length]
    return paragragh_list