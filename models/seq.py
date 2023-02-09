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


class SeqCls(nn.Module):
    def __init__(self, nclass:int, model_name:str, **kwargs):
        super().__init__()
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.event_cls = nn.Linear(d_model, nclass)
        self.crit = nn.CrossEntropyLoss()

    def compute_cross_entropy(self, logits, labels, crit=None):
        mask = labels >= 0
        if crit is None: crit = self.crit
        return crit(logits[mask], labels[mask])
    
    def soft_binary(self, logits, labels):
        return torch.mean(- labels * torch.log_softmax(logits, dim=-1))

    def forward(self, batch):
        encoded = self.pretrained_lm(batch["input_ids"], batch["attention_mask"])
        outputs = self.event_cls(encoded.last_hidden_state)
        
        labels = batch["trigger_labels"]
        loss = self.compute_cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=-1)
        preds[labels < 0] = labels[labels < 0]
        labels = labels

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
        self.event_crf = SimpleCRFHead(nstate=nclass["event"])
        # self.relation_cls = Bilinear(d_model, d_model, nclass["relation"])
        self.relation_cls = BiClassifier(d_model, d_model, nclass["relation"])
        self.role_cls = Bilinear(d_model, d_model, nclass["role"])
        self.role_event_cls = Bilinear(d_model, d_model, nclass["event"])

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