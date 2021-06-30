import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from typing import *
import warnings
import math

class TransformerLM(nn.Module):
    def __init__(self, d_model:int, nhead:int, num_layers:int, max_length:int, input_dropout:float=0.1, model_dropout:float=0.1, vocab_size:int=0, blk:int=-1, sos:int=-1, eos:int=-1, use_embedding:bool=True, use_pred:bool=True, no_gpu:bool=False) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=model_dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None
        if use_pred:  
            self.vocab_pred = nn.Linear(d_model, vocab_size, bias=True)
        else:
            self.vocab_pred = None
        self.position = torch.zeros(max_length, d_model, requires_grad=False)
        position = torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() / d_model * math.log(1e-4))
        self.position[:, 0::2] = torch.sin(position * div_term)
        self.position[:, 1::2] = torch.cos(position * div_term)
        self.position = nn.Parameter(self.position.unsqueeze(0))
        self.input_dropout = nn.Dropout(input_dropout)
        if not no_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.blk = 0 if blk == -1 else blk
        self.eos = vocab_size - 1 if eos == -1 else eos
        self.sos = vocab_size - 2 if sos == -1 else sos
        self.vocab_size = vocab_size
    
    def forward(self, seq:torch.LongTensor, memory:torch.FloatTensor, tgt_mask:Union[torch.FloatTensor, None]=None, tgt_key_padding_mask:Union[torch.BoolTensor, None]=None, memory_mask:Union[torch.FloatTensor, None]=None, memory_key_padding_mask:Union[torch.BoolTensor, None]=None, embedding:Union[nn.Embedding, None]=None, vocab_pred:Union[nn.Module, None]=None, return_hidden:bool=True):
        if embedding is not None and self.embedding is not None:
            warnings.warn("Using externel embeddings for <function: TransformerLM.forward>.")
        if embedding is None:
            assert self.embedding is not None, "Need to provide embedding for a TransformerLM initialized with 'use_embedding=False'"
            embedding = self.embedding
        if vocab_pred is not None and self.vocab_pred is not None:
            warnings.warn("Using externel LM heads for <function: TransformerLM.forward>.")
        if vocab_pred is None:
            if self.vocab_pred is None:
                warnings.warn("LM heads not provided for a TransformerLM initialized with 'use_pred=False', returning only hidden_layer")
            vocab_pred = self.vocab_pred
        bsz = seq.size(0)
        seq_length = seq.size(1)
        assert memory.size(1)==bsz, "Please put memory in shape 'length x batch_size x dim'"
        if tgt_mask is None:
            tgt_mask = self.generate_mask(seq_length)
        target = self.input_dropout(embedding(seq) + self.position[:, :seq_length, :]).transpose(0, 1)
        memory = self.input_dropout(memory)
        output = self.decoder(target, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask).transpose(0, 1)
        if vocab_pred:
            logits = vocab_pred(self.input_dropout(output))
            if return_hidden:
                return logits, output
            else:
                return logits
        else:
            return output
    
    def forward_step(self, inputs:torch.FloatTensor, memory:torch.FloatTensor, memory_key_padding_mask:Union[torch.BoolTensor, None]=None) -> torch.FloatTensor:
        # inputs: B x S-1 x D
        # memory: B x T x D
        # memory_key_padding_mask: B x T
        inputs = inputs + self.position[:, :inputs.size(1), :]
        tgt_mask = self.generate_mask(inputs.size(1))
        output = self.decoder(inputs.transpose(0, 1), memory.transpose(0, 1), tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask).transpose(0, 1)
        return output[:, -1:, :]
    
    def decode_beam(self, memory:torch.FloatTensor, memory_key_padding_mask:Union[torch.BoolTensor, None]=None, embedding:Union[nn.Embedding, None]=None, vocab_pred:Union[nn.Module, None]=None, max_len:int=-1, beam:int=10):
        if max_len == -1:
            max_len = self.position.size(1)
        elif max_len > self.position.size(1):
            warnings.warn("Decoding to length larger than TransformerLM's maximum length. Reset to maximum length.")
            max_len = self.position.size(1)
        if embedding is not None and self.embedding is not None:
            warnings.warn("Using externel embeddings for <function: TransformerLM.forward>.")
        if embedding is None:
            assert self.embedding is not None, "Need to provide embedding for a TransformerLM initialized with 'use_embedding=False'"
            embedding = self.embedding
        if vocab_pred is not None and self.vocab_pred is not None:
            warnings.warn("Using externel LM heads for <function: TransformerLM.forward>.")
        if vocab_pred is None:
            if self.vocab_pred is None:
                warnings.warn("LM heads not provided for a TransformerLM initialized with 'use_pred=False', returning only hidden_layer")
            vocab_pred = self.vocab_pred
        bs = memory.size(0)
        sos = torch.ones(size=(bs, 1), dtype=torch.long, device=self.device) * self.sos
        next_input = embedding(sos)
        prob = torch.zeros(size=(bs, beam, 1), dtype=torch.float, device=self.device)
        outputs = torch.zeros(size=(bs, beam, max_len+1), dtype=torch.long, device=self.device)
        outputs[:, :, 0] = self.sos
        final_outputs = [[] for _ in range(bs)]
        final_prob = [[] for _ in range(bs)]
        finished = torch.zeros((bs,), dtype=torch.bool, device=self.device)
        # first_step
        output = self.forward_step(next_input, memory, memory_key_padding_mask)
        pred = torch.log_softmax(vocab_pred(output), dim=-1)
        val, ind = pred.topk(k=beam, dim=-1)
        ind = ind.view(bs, beam, 1)
        val = val.view(bs, beam, 1)
        next_step_input = embedding(ind)
        next_input = torch.cat((next_input.repeat_interleave(beam, dim=0),
                                next_step_input.flatten(end_dim=1)), dim=1)
        prob = prob + val
        memory = memory.repeat_interleave(beam, dim=0)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(beam, dim=0)
        outputs[:, :, 1] = ind.squeeze(-1)
        for i in range(1, max_len):
            output = self.forward_step(next_input, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
            # output bs*beam x 1 x d, hidden bs*beam x N x d
            pred = torch.log_softmax(vocab_pred(output), dim=-1)
            # pred bs*beam x 1 x v_size
            val, ind = pred.topk(k=beam, dim=-1)
            # val, ind bs*beam x 1 x beam
            val = val.view(bs, beam, beam)
            val = val + prob
            val = val.view(bs, beam * beam)
            prob, new_beam = val.topk(k=beam, dim=-1)
            # prob, new_beam bs x beam
            prob = prob.unsqueeze(-1)
            # prob, bs x beam x 1
            ind = ind.view(bs, beam * beam)
            ind = torch.gather(ind, dim=1, index=new_beam)
            outputs = outputs.repeat_interleave(beam, dim=1)
            outputs = torch.gather(outputs, dim=1, index=new_beam.unsqueeze(-1).repeat_interleave(outputs.size(-1), dim=-1))
            outputs[:, :, i+1] = ind
            next_input = embedding(outputs[:, :, :i+2]).flatten(end_dim=1)
            ending = (ind==self.eos).long()
            ending_num = torch.sum(ending, dim=1)
            finished = finished + (ind[:, 0].flatten() == self.eos)
            terminate = all(finished)
            for j in range(ind.size(0)):
                if ending_num[j] == 0:
                    continue
                final_outputs[j].extend(outputs[j][ind[j]==self.eos][:, 1:i+1].tolist())
                final_prob[j].extend(prob[j][ind[j]==self.eos].squeeze(-1).tolist())
                prob[j][ind[j]==self.eos] = float("-inf")
            if terminate:
                break
        for i in range(bs):
            if len(final_outputs[i]) == 0:
                final_outputs[i] = outputs[i, :, 1:].tolist()
                final_prob[i] = prob[i].squeeze(-1).tolist()
            best_index = final_prob[i].index(max(final_prob[i]))
            final_outputs[i] = final_outputs[i][best_index]
        return final_outputs

    def generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)


class IESPAN(nn.Module):
    def __init__(self, hidden_dim:int, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.linear_map = nn.Linear(2 * d_model, nclass)
        self.dropout = nn.Dropout(0.0)
        self.label_info = None
        self.train_labels = -1
        self.crit = nn.CrossEntropyLoss()

    def forward(self, batch):
        token_ids, attention_masks, spans, labels = batch["input_ids"], batch["attention_mask"], batch['spans'], batch["labels"]
        encoded = self.pretrained_lm(token_ids, attention_masks, output_hidden_states=True)
        encoded = torch.cat((encoded.last_hidden_state, encoded.hidden_states[-3]), dim=-1)
        candidates = torch.matmul(spans, encoded)
        span_mask = labels >= 0

        outputs = self.linear_map(candidates)
        loss = self.crit(outputs[span_mask], labels[span_mask])
        preds = torch.argmax(outputs, dim=-1)

        ngold = torch.sum((labels > 0)[span_mask].float())
        npred = torch.sum((preds > 0)[span_mask].float())
        match = torch.sum(((preds == labels)*(labels > 0))[span_mask].float())

        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach(),
            "f1": torch.tensor([ngold, npred, match])
            }

class CrossIE(nn.Module):
    def __init__(self, hidden_dim:int, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.event_att = nn.Linear(d_model*2, nclass)
        self.embeddings = self.pretrained_lm.get_input_embeddings()
        self.generator = TransformerLM(
            d_model=self.embeddings.embedding_dim,
            nhead=8,
            num_layers=3,
            max_length=64,
            input_dropout=0.1,
            model_dropout=0.0,
            vocab_size=self.embeddings.num_embeddings,
            blk=tokenizer.pad_token_id,
            sos=tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id,
            eos=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id,
            use_embedding=False,
            use_vocab_pred=False)

        
        self.vocab_pred = nn.Linear(self.embeddings.embedding_dim, self.embeddings.num_embeddings, bias=False)
        self.vocab_pred.weight.data = self.embeddings.weight.data.detach().clone()
        self.vocab_pred.weight.requires_grad = False
        self.crit = nn.CrossEntropyLoss()
        self.dist_crit = nn.SmoothL1Loss(reduction='mean')
        self.outputs = dict()

    def compute_cross_entropy(self, logits, labels):
        mask = labels >= 0
        return self.crit(logits[mask], labels[mask])

    def forward(self, batch):
        token_ids, attention_masks, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        encoded = self.pretrained_lm(token_ids, attention_masks, output_hidden_states=True)
        encoded = torch.cat((encoded.last_hidden_state, encoded.hidden_states[-3]), dim=-1)
        event_att_v = torch.matmul(torch.softmax(self.event_att(encoded), dim=-1).transpose(1, 2), encoded)
        
        # extraction loss
        outputs = torch.matmul(encoded, event_att_v.transpose(1, 2))
        loss = self.compute_cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        preds[labels < 0] = labels[labels < 0]

        # generation loss
        reconstruct = self.generator.forward(seq=batch["input_ids"], memory=event_att_v.transpose(0, 1), embedding=self.embeddings.detach(), vocad_pred=self.vocab_pred, return_hidden=False)
        reconstruct = reconstruct[:, :-1, :]
        reconstruct_label = batch["input_ids"][:, 1:, :]
        reconstruct_loss = self.compute_cross_entropy(reconstruct, reconstruct_label)

        # distance loss
        distance_loss = 0.
        if batch["parallel"] is not None:
            parallel = torch.index_select(event_att_v, 0, batch["parallel"]).view(batch["parallel"].size(0) // 2, 2, -1)
            distance_loss = self.dist_crit(parallel[:, 0, :], parallel[:, 1, :])
            

        return {
            "loss": loss + reconstruct_loss + distance_loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }