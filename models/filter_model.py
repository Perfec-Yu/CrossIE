from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaLayer, RobertaSelfOutput, RobertaAttention, RobertaOutput, RobertaEncoder, RobertaPooler

from transformers_filter import FilterModel, AutoConfig
import torch
import torch.nn as nn

class _FilterModel(nn.Module):
    def __init__(self, filter_m:int, filter_k:int):
        super().__init__()
        config = AutoConfig.from_pretrained("xlm-roberta-large")
        config.filter_m = filter_m; config.filter_k = filter_k; config.output_hidden_states = True
        self.model = FilterModel.from_pretrained("xlm-roberta-large", config=config)
    
    def forward(self, inputs):#*inputs_langs):
        # inputs = {
        #     key: [lang[key] for lang in inputs_langs] for key in inputs_langs[0]
        # }
        # lens = [t['inputs_ids'].size(1) for t in inputs]
        lens = [t.size(1) for t in inputs['input_ids']]
        outputs = self.model(**inputs)
        outputs_langs = []
        last = 0
        for lang_len in lens:
            outputs_langs.append(outputs[:, last:last+lang_len])
            last += lang_len
        return outputs_langs