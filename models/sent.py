import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from typing import *
import warnings
import math
from .utils import SimpleCRFHead, Bilinear
from .filter_model import _FilterModel

class SentCls(nn.Module):
    def __init__(self, nclass:int, model_name:str, loss_type='sigmoid', **kwargs):
        super().__init__()
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.loss_type = loss_type
        self.role_cls = nn.Linear(d_model, nclass-1 if loss_type == 'sigmoid' else nclass)
        self.crit = nn.BCEWithLogitsLoss() if loss_type == 'sigmoid' else nn.CrossEntropyLoss()

    def compute_cross_entropy(self, logits, labels, crit=None):
        mask = labels >= 0
        if crit is None: crit = self.crit
        return crit(logits[mask], labels[mask])

    def forward(self, batch):
        encoded = self.pretrained_lm(batch["input_ids"], batch["attention_mask"])
        outputs = self.role_cls(encoded.last_hidden_state[:, 0, :])
        if self.loss_type == 'sigmoid':
            preds = outputs > 0
        else:
            preds = torch.argmax(outputs, dim=-1)
        loss = self.compute_cross_entropy(outputs, batch['sentence_labels'])

        return {
            "loss": loss,
            "prediction": preds.long().detach(),
            "label": batch['sentence_labels'].long().detach()
            }


class SentEDFilter(nn.Module):
    def __init__(self, hidden_dim:int, nclass:Union[Dict, int], filter_m:int, filter_k:int, classifier:str='sigmoid', task_str:str='event', **kwargs):
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
        self.dim_map = nn.Linear(d_model, self.hidden_dim) if self.hidden_dim != d_model else None
        self.classifier = classifier

        print(nclass[task_str])
        if classifier == 'sigmoid':
            self.crit = nn.BCEWithLogitsLoss()
            self.event_cls = nn.Linear(d_model*1, nclass[task_str]-1)
        else:
            self.crit = nn.CrossEntropyLoss()
            self.event_cls = nn.Linear(d_model*1, nclass[task_str])

    def soft_binary(self, logits, labels):
        return torch.mean(- labels * torch.nn.functional.logsigmoid(logits) + (labels - 1) * torch.nn.functional.logsigmoid(-logits))

    def compute_cross_entropy(self, logits, labels, crit=None):
        mask = labels >= 0
        if torch.sum(mask) == 0:
            return torch.tensor(0., device=logits.device)
        if crit is None: crit = self.crit
        return crit(logits[mask], labels[mask])


    def forward(self, batch):
        lm_inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        encoded_langs = self.pretrained_lm(lm_inputs)
        encoded_langs = [encoded[:, 0, :] for encoded in encoded_langs]
        
        outputs = [self.event_cls(encoded if self.dim_map is None else self.dim_map(encoded)) for encoded in encoded_langs]
        if self.classifier == 'sigmoid':
            preds = [output > 0 for output in outputs]
        else:
            preds = [torch.argmax(output, dim=-1) for output in outputs]
        losses = [self.compute_cross_entropy(output, batch['sentence_labels'][batch["eval_lang"] if "eval_lang" in batch else 0]) for i, output in enumerate(outputs)]
        loss = sum(losses)
        if "eval_lang" in batch:
            pred = preds[batch["eval_lang"]]
            label = batch['sentence_labels'][batch["eval_lang"]]
        else:
            pred = torch.cat(preds, dim=0)
            label = torch.cat(batch['sentence_labels'], dim=0)

        return {
            "loss": loss,
            "prediction": pred.long().detach().cpu(),
            "label": label.long().detach().cpu(),
            "outputs": encoded_langs[batch["eval_lang"]].detach().cpu()
            }