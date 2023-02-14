import transformers
import torch

class biEncoder(torch.nn.Module):
    def __init__(self, CONF: dict):
        super(biEncoder, self).__init__()
        self.passage_encoder = transformers.BertModel.from_pretrained(CONF['model']['name'])
        self.query_encoder = transformers.BertModel.from_pretrained(CONF['model']['name'])
        # self.embedding_size = self.passage_encoder.pooler.dense.out_features

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, isPassage: bool) -> torch.FloatTensor:
        return self.passage_encoder(input_ids = input_ids, attention_mask = attention_mask).pooler_output if isPassage else \
                self.query_encoder(input_ids = input_ids, attention_mask = attention_mask).pooler_output