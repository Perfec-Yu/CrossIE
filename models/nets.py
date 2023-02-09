import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from typing import *
import warnings
import math
from .utils import SimpleCRFHead, Bilinear, BiClassifier
from .filter_model import _FilterModel

def _span_representations(batch_of_sequences:torch.Tensor, span_masks:torch.Tensor):
    '''
    batch_of_sequences: B x L x d
    span_masks: B x N x L 
    return: B x N x d
    '''
    return torch.matmul(span_masks, batch_of_sequences)


def _spans(batch_of_spans:torch.Tensor, index:torch.LongTensor):
    '''
    batch_of_spans: B x N x d
    index: M x 2, Sequence of [batch_index, sequence_index]
    return: M x d
    '''
    return batch_of_spans[index[:, 0], index[:, 1]]


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


class CrossIE(nn.Module):
    def __init__(self, hidden_dim:int, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        nclass = nclass['event']
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 768)
        self.event_att = nn.Linear(d_model*1, 1)
        self.event_sent_cls = nn.Linear(d_model*1, nclass-1)
        self.event_cls = nn.Linear(d_model*1, nclass)

        self.projection = lambda t:t #nn.Linear(d_model*2, d_model)
        self.embeddings = nn.Embedding.from_pretrained(self.pretrained_lm.get_input_embeddings().weight.data.detach().clone(), freeze=True)
        self.generator = TransformerLM(
            d_model=self.embeddings.embedding_dim,
            nhead=8,
            num_layers=3,
            max_length=96,
            input_dropout=0.1,
            model_dropout=0.0,
            vocab_size=self.embeddings.num_embeddings,
            blk=tokenizer.pad_token_id,
            sos=tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.cls_token_id,
            eos=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id,
            use_embedding=False,
            use_pred=False)

        
        self.vocab_pred = nn.Linear(self.embeddings.embedding_dim, self.embeddings.num_embeddings, bias=False)
        self.vocab_pred.weight.data = self.embeddings.weight.data.detach().clone()
        self.vocab_pred.weight.requires_grad = False

        self.crit = nn.CrossEntropyLoss()
        self.sent_crit = nn.BCEWithLogitsLoss()
        self.dist_crit = nn.MSELoss(reduction='mean')
        self.outputs = dict()

    def compute_cross_entropy(self, logits, labels, crit=None):
        mask = labels >= 0
        if crit is None: crit = self.crit
        return crit(logits[mask], labels[mask])

    def forward(self, batch):
        token_ids, attention_masks, labels = batch["input_ids"], batch["attention_mask"], batch["trigger_labels"]
        encoded = self.pretrained_lm(token_ids, attention_masks, output_hidden_states=True)
        encoded = encoded.last_hidden_state#torch.cat((encoded.last_hidden_state, encoded.hidden_states[-3]), dim=-1)
        event_att_v = torch.matmul(torch.softmax(self.event_att(encoded), dim=-2).transpose(1, 2), encoded)
        event_att_v = encoded[:, :1, :]
        sent_outputs = self.event_sent_cls(event_att_v).squeeze(1)
        outputs = self.event_cls(encoded)
        
        # extraction loss
        loss = self.compute_cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        preds[labels < 0] = labels[labels < 0]

        # sentence loss
        sent_loss = self.compute_cross_entropy(sent_outputs, batch['sentence_labels'], self.sent_crit)

        # generation loss
        reconstruct = self.generator.forward(seq=batch["input_ids"], memory=self.projection(event_att_v.transpose(0, 1)), embedding=self.embeddings, vocab_pred=self.vocab_pred, return_hidden=False)
        reconstruct = reconstruct[:, :-1, :]
        reconstruct_label = batch["input_ids"][:, 1:]
        reconstruct_loss = self.compute_cross_entropy(reconstruct, reconstruct_label)

        # distance loss
        distance_loss = 0.
        if "parallel" in batch and batch["parallel"] is not None:
            parallel = torch.index_select(event_att_v, 0, batch["parallel"]).view(batch["parallel"].size(0) // 2, 2, -1)
            distance_loss = self.dist_crit(parallel[:, 0, :], parallel[:, 1, :])

        return {
            "loss": loss, #+ sent_loss +  1e-4 * distance_loss,#+ reconstruct_loss ,#+,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }


class SupIE(nn.Module):
    def __init__(self, hidden_dim:int, nclass:Union[Dict, int], model_name:str, **kwargs):
        super().__init__()
        if isinstance(nclass, int):
            nclass = {
                "event": nclass,
                "relation": nclass,
                "role": nclass
            }
        self.hidden_dim = hidden_dim
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.event_cls = nn.Linear(d_model*1, nclass["event"])
        self.event_crf = SimpleCRFHead(nstate=nclass["event"])
        self.relation_cls = BiClassifier(d_model, d_model, nclass["relation"])
        self.role_cls = nn.Linear(d_model, d_model, nclass["role"])
        self.role_event_cls = Bilinear(d_model, d_model, nclass["event"])

        self.projection = lambda t:t #nn.Linear(d_model*2, d_model)

        self.crit = nn.CrossEntropyLoss()
        self.outputs = dict()

    def compute_cross_entropy(self, logits, labels, crit=None):
        mask = labels >= 0
        if crit is None: crit = self.crit
        return crit(logits[mask], labels[mask])
    
    def soft_binary(self, logits, labels):
        return torch.mean(- labels * torch.log_softmax(logits, dim=-1))
    
    def event_forward(self, batch, encoded=None):
        if encoded is None:
            encoded = self.pretrained_lm(batch["input_ids"], batch["attention_mask"])
            encoded = encoded.last_hidden_state
        outputs = self.event_cls(encoded)
        loss = self.compute_cross_entropy(outputs, batch["trigger_labels"])
        preds = torch.argmax(outputs, dim=-1)
        preds[batch["trigger_labels"] < 0] = batch["trigger_labels"][batch["trigger_labels"] < 0]
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": batch["trigger_labels"].long().detach()
            }
    
    def relation_forward(self, batch, encoded=None):
        if encoded is None:
            encoded = self.pretrained_lm(batch["input_ids"], batch["attention_mask"])
            encoded = encoded.last_hidden_state
        entity_repr = _span_representations(encoded, batch["entity_spans"])
        relation_outputs = self.relation_cls(entity_repr, entity_repr)
        loss = self.compute_cross_entropy(relation_outputs, batch["relation_labels"])
        relation_outputs = relation_outputs[batch["relation_labels"] >= 0]
        relation_labels = batch["relation_labels"][batch["relation_labels"] >= 0]
        labels = relation_labels
        preds = torch.argmax(relation_outputs, dim=1)
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }
    

    def role_forward(self, batch, encoded=None):
        if encoded is None:
            encoded = self.pretrained_lm(batch["input_ids"], batch["attention_mask"])
            encoded = encoded.last_hidden_state
        entity_repr = _span_representations(encoded, batch["entity_spans"])
        trigger_repr = _span_representations(encoded, batch["trigger_spans"])
        role_outputs = self.role_cls(trigger_repr, entity_repr)
        loss = self.compute_cross_entropy(role_outputs, batch["role_labels"])
        role_outputs = role_outputs[batch["role_labels"] >= 0]
        role_labels = batch["role_labels"][batch["role_labels"] >= 0]
        labels = role_labels
        preds = torch.argmax(role_outputs, dim=1)
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }

    def forward(self, batch):
        token_ids, attention_masks, trigger_labels = batch["input_ids"], batch["attention_mask"], batch["trigger_labels"]
        encoded = self.pretrained_lm(token_ids, attention_masks, output_hidden_states=True)
        encoded = encoded.last_hidden_state
        
        # event detection
        outputs = self.event_cls(encoded)
        loss = self.compute_cross_entropy(outputs, trigger_labels)
        preds = torch.argmax(outputs, dim=-1)
        preds[trigger_labels < 0] = trigger_labels[trigger_labels < 0]
        labels = trigger_labels

        # # relation extraction
        # entity_repr = _span_representations(encoded, batch["entity_spans"])
        # # entity_head = _spans(entity_repr, batch["relation_pairs"][:, :2])
        # # entity_tail = _spans(entity_repr, batch["relation_pairs"][:, 0::2])
        # relation_outputs = self.relation_cls(entity_repr, entity_repr)
        # loss_relation = self.compute_cross_entropy(relation_outputs, batch["relation_labels"])

        # # trigger_repr = _span_representations(encoded, batch["trigger_spans"])
        # # role_trigger = _spans(trigger_repr, batch["role_pairs"][:, :2])
        # # role_argument = _spans(entity_repr, batch["role_pairs"][:, 0::2])
        # # role_outputs = self.role_cls(trigger_repr, entity_repr)
        # # role_event_outputs = self.role_event_cls(trigger_repr, entity_repr)
        # # loss_role = self.compute_cross_entropy(role_outputs, batch["role_labels"])
        # # loss_role_event = self.compute_cross_entropy(role_event_outputs, batch["role_event_labels"])

        # relation_outputs = relation_outputs[batch["relation_labels"] >= 0]
        # relation_labels = batch["relation_labels"][batch["relation_labels"] >= 0]
        # # role_outputs = role_outputs[batch["role_labels"] >= 0]
        # # role_event_outputs = role_event_outputs[batch["role_labels"] >= 0]
        # # role_labels = batch["role_labels"][batch["role_labels"] >= 0]
        # # role_event_labels = batch["role_labels"][batch["role_labels"] >= 0]
        # # scoring_labels = role_labels #+ role_event_labels * 100
        # # scoring_predictions = torch.argmax(role_outputs, dim=1) #+ torch.argmax(role_event_outputs, dim=1) * 100
        # # scoring_labels = relation_labels
        # # scoring_predictions = 
        # labels = relation_labels
        # preds = torch.argmax(relation_outputs, dim=1)

        # # loss = loss_role# + loss_role_event
        # loss = loss_relation

        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }


class SupFilterIE(nn.Module):
    def __init__(self, hidden_dim:int, nclass:Union[Dict, int], filter_k:int, filter_m:int, **kwargs):
        super().__init__()
        if isinstance(nclass, int):
            nclass = {
                "event": nclass,
                "relation": nclass,
                "role": nclass
            }
        self.hidden_dim = hidden_dim
        self.pretrained_lm = _FilterModel(filter_k=filter_k, filter_m=filter_m)
        d_model = 1024
        self.event_cls = nn.Linear(d_model, nclass["event"])
        # self.event_crf = SimpleCRFHead(nstate=nclass["event"])
        # self.relation_cls = Bilinear(d_model, d_model, nclass["relation"])
        # self.relation_cls = BiClassifier(d_model, d_model, nclass["relation"])
        # self.role_cls = Bilinear(d_model, d_model, nclass["role"])
        # self.role_event_cls = Bilinear(d_model, d_model, nclass["event"])

        self.dim_map = nn.Linear(d_model, hidden_dim) if d_model != hidden_dim else None

        self.crit = nn.CrossEntropyLoss()
        self.outputs = dict()
    
    def soft_entropy(self, logits, labels):
        return torch.mean(- labels * torch.log_softmax(logits, dim=-1))
    
    def compute_cross_entropy(self, logits, labels, crit=None):
        mask = labels >= 0
        if torch.sum(mask) == 0: return torch.tensor(0., device=logits.device)
        if crit is None: crit = self.crit
        return crit(logits[mask], labels[mask])
    
    def get_encoded(self, batch):
        lm_inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        encoded_langs = self.pretrained_lm(lm_inputs)
        outputs = [encoded if self.dim_map is None else self.dim_map(encoded) for encoded in encoded_langs]
        return outputs
    
    def get_encoded_relation(self, batch):
        encoded = self.get_encoded(batch)
        return [_span_representations(_encoded, _span) for _encoded, _span in zip(encoded, batch["entity_spans"])]
        
    def event_forward(self, batch=None, encoded=None, label=None, self_training=False, loss_lang=None):
        if encoded is None:
            assert batch is not None
            encoded = self.get_encoded(batch)
        if label is None:
            assert self_training or batch is not None
            label = batch["trigger_labels"]
        outputs = [self.event_cls(_encoded) for _encoded in encoded]
        crit = self.soft_entropy if self_training else self.crit
        if loss_lang is None:
            if self_training:
                loss = sum([self.compute_cross_entropy(o, torch.softmax(o*2, dim=-1).detach(), crit=crit) for o in outputs])
            else:
                loss = sum([self.compute_cross_entropy(o, l, crit=crit) for o, l in zip(outputs, label)])
        else:
            loss = self.compute_cross_entropy(outputs[loss_lang], label[loss_lang])
        output_idx = loss_lang if loss_lang else 0
        preds = torch.argmax(outputs[output_idx], dim=-1)
        preds[label[output_idx] < 0] = label[output_idx][label[output_idx] < 0]
        return {
            "loss": loss,
            "prediction": preds.long().detach().cpu(),
            "label": label[output_idx].long().detach().cpu()
            }
    
    def relation_forward(self, batch=None, encoded=None, label=None, self_training=False, loss_lang=None):
        if encoded is None:
            encoded = self.get_encoded_relation(batch)
        if label is None:
            assert self_training or batch is not None
            label = batch["relation_labels"]
        entity_repr = encoded
        relation_outputs = [self.relation_cls(_entity_repr, _entity_repr) for _entity_repr in entity_repr]
        if loss_lang is None:
            loss = sum([self.compute_cross_entropy(relation_output, relation_label) for relation_output, relation_label in zip(relation_outputs, label)])
        else:
            loss = self.compute_cross_entropy(relation_outputs[loss_lang], label[loss_lang])
        output_idx = loss_lang if loss_lang else 0
        relation_outputs = relation_outputs[output_idx][label[output_idx] >= 0]
        relation_labels = label[output_idx][label[output_idx] >= 0]
        labels = relation_labels
        preds = torch.argmax(relation_outputs, dim=1)
        if output_idx != 1:
            print(output_idx)
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }
    

    def role_forward(self, batch, encoded=None):
        if encoded is None:
            encoded = self.get_encoded(batch)
        entity_repr = _span_representations(encoded, batch["entity_spans"])
        trigger_repr = _span_representations(encoded, batch["trigger_spans"])
        role_outputs = self.role_cls(trigger_repr, entity_repr)
        loss = self.compute_cross_entropy(role_outputs, batch["role_labels"])
        role_outputs = role_outputs[batch["role_labels"] >= 0]
        role_labels = batch["role_labels"][batch["role_labels"] >= 0]
        labels = role_labels
        preds = torch.argmax(role_outputs, dim=1)
        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": labels.long().detach()
            }
    
    def forward(self, batch):
        return self.event_forward(batch, loss_lang=None if "eval_lang" not in batch else batch["eval_lang"])