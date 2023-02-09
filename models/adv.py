import torch
import torch.nn as nn
from .sent import SentEDFilter
from .nets import SupIE, SupFilterIE, _span_representations
from typing import Union, Dict

class Adv(nn.Module):
    def __init__(self, hidden_dim:int, _lambda:float=1., weight_cliping_limit=0.01):
        super(Adv, self).__init__()
        self.adv_d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._lambda = _lambda
        self.weight_cliping_limit = weight_cliping_limit

    def clip_params(self,):
        for p in self.adv_d.parameters():
            p.data.clamp(-self.weight_cliping_limit, self.weight_cliping_limit)

    def _g_forward(self, features_t, features_n):
        outputs_t, outputs_n = self.adv_d(features_t), self.adv_d(features_n)
        loss_g = - outputs_t.mean() + outputs_n.mean()
        return loss_g

    def _d_forward(self, features_t, features_n):
        outputs_t, outputs_n = self.adv_d(features_t.detach()), self.adv_d(features_n.detach())
        loss_d = outputs_t.mean() - outputs_n.mean() + self._gradient_penalty(features_t, features_n)
        return loss_d

    def _gradient_penalty(self, features_t, features_n):
        if features_t.size(0) != features_n.size(0):
            if features_t.size(0) < features_n.size(0):
                taken_size = features_t.size(0)
                sample_ind = torch.randperm(features_n.size(0), device=features_n.device)[:taken_size]
                s_features_t = features_t
                s_features_n = features_n[sample_ind]
            else:
                taken_size = features_n.size(0)
                sample_ind = torch.randperm(features_t.size(0), device=features_t.device)[:taken_size]
                s_features_t = features_t[sample_ind]
                s_features_n = features_n
        else:
            s_features_t = features_t
            s_features_n = features_n
        epsilon = torch.rand([s_features_n.size(0)] + [1] * (len(s_features_n.size()) - 1), device=s_features_n.device)
        s_features = epsilon * s_features_t + (1 - epsilon) * s_features_n
        s_features.requires_grad_(True)
        s_outputs = self.adv_d(s_features)

        grads = torch.autograd.grad(outputs=s_outputs, inputs=s_features, grad_outputs=torch.ones_like(s_outputs), create_graph=True)[0]
        grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean() * self._lambda
        return grad_penalty

    def forward(self, features_t, features_n, mode):
        if mode == "g":
            return self._g_forward(features_t, features_n)
        else:
            return self._d_forward(features_t, features_n)


class SentEDFilterAdv(nn.Module):
    def __init__(self, nclass:Union[Dict, int], model_name:str, filter_m:int, filter_k:int, hidden_dim=1024, _lambda=10., _g_lambda=1e-2, classifier:str='sigmoid', task_str:str='event', **kwargs):
        super().__init__()
        self.adv_g = SentEDFilter(hidden_dim=hidden_dim, nclass=nclass, filter_m=filter_m, filter_k=filter_k, classifier=classifier, task_str=task_str)
        self.adv_c_t = Adv(hidden_dim, _lambda)
        self.adv_c_s = Adv(hidden_dim, _lambda)
        self._g_lambda = _g_lambda

    def clip_params(self,):
        self.adv_c_s.clip_params()
        self.adv_c_t.clip_params()

    def get_d_params(self,):
        return list(self.adv_c_s.parameters()) + list(self.adv_c_t.parameters())

    def get_g_params(self,):
        return list(self.adv_g.parameters())

    def get_features(self, batch_l, batch_u):
        lm_inputs_l = {"input_ids": batch_l["input_ids"], "attention_mask": batch_l["attention_mask"]}
        encoded_langs_l = [encoded[:, 0, :] if self.adv_g.dim_map is None else self.adv_g.dim_map(encoded[:, 0, :]) for encoded in self.adv_g.pretrained_lm(lm_inputs_l)]

        lm_inputs_u = {"input_ids": batch_u["input_ids"], "attention_mask": batch_u["attention_mask"]}
        encoded_langs_u = [encoded[:, 0, :] if self.adv_g.dim_map is None else self.adv_g.dim_map(encoded[:, 0, :]) for encoded in self.adv_g.pretrained_lm(lm_inputs_u)]

        n_l = encoded_langs_l[0]; t_l = encoded_langs_l[1]
        n_u = encoded_langs_u[1]; t_u = encoded_langs_u[0]
        return (t_l, n_u), (t_u, n_l)

    def _task_step(self, feature_ln, feature_lt, labels, copy_labels=True, eval_lang=None, self_training=False, return_outputs=False):
        outputs = [self.adv_g.event_cls(t) for t in [feature_ln, feature_lt]]
        preds = [output > 0 for output in outputs]
        label_lang = eval_lang if eval_lang else 0
        if self_training:
            losses = [self.compute_cross_entropy(output, torch.sigmoid(output*2).detach(), crit=self.adv_g.soft_binary) for i, output in enumerate(outputs)]
        else:
            losses = [self.compute_cross_entropy(output, labels[label_lang if copy_labels else i]) for i, output in enumerate(outputs)]

        loss = sum(losses)
        if return_outputs:
            if eval_lang:
                pred = preds[eval_lang]
                label = labels[eval_lang]
            else:
                pred = torch.cat(preds, dim=0)
                label = torch.cat(labels, dim=0)

            return {
                "loss": loss,
                "prediction": pred.long().detach(),
                "label": label.long().detach()
                }
        else:
            return loss

    def d_forward(self, features_t, features_s):
        return self.adv_c_t(*features_t, mode="d") + self.adv_c_s(*features_s, mode="d")

    def g_forward(self, batch_l, batch_u, features_t=None, features_s=None, warmup=False, self_training=False):
        if features_t is None or features_s is None:
            features_t, features_s = self.get_features(batch_l, batch_u)
        eval_lang_l = None if "eval_lang" not in batch_l else batch_l["eval_lang"]
        outputs = self._task_step(
            feature_ln=features_s[1],
            feature_lt=features_t[0],
            labels=batch_l["sentence_labels"],
            copy_labels=True,
            eval_lang=eval_lang_l,
            return_outputs=True)
        if self_training:
            outputs["loss"] += 0.1 * self._task_step(
                feature_ln=features_t[1],
                feature_lt=features_s[0],
                labels=None,
                copy_labels=False,
                eval_lang=None,
                self_training=True,
                return_outputs=False)
        elif "pseudo_sentence_labels" in batch_u:
            outputs["loss"] += self._task_step(
                feature_ln=features_t[1],
                feature_lt=features_s[0],
                labels=batch_u["pseudo_sentence_labels"],
                copy_labels=False,
                eval_lang=None,
                return_outputs=False)

        if not warmup:
            g_loss = self.adv_c_t(*features_t, mode="g") + self.adv_c_s(*features_s, mode="g")
            outputs["g_loss"] = self._g_lambda * g_loss
        return outputs

    def compute_cross_entropy(self, logits, labels, crit=None):
        return self.adv_g.compute_cross_entropy(logits, labels, crit=crit)

    def forward(self, batch_l=None, batch_u=None, features_t=None, features_s=None, mode="g", warmup=False, self_training=False):
        '''
            mode = ['g', 'd', 'eval', 'feat']
        '''
        if mode == 'eval':
            return self.adv_g.forward(batch_l)
        elif mode == "feat":
            return self.get_features(batch_l, batch_u)
        elif mode == "g":
            return self.g_forward(batch_l, batch_u, features_t, features_s, warmup=warmup, self_training=self_training)
        elif mode == "d":
            return self.d_forward(features_t, features_s)
        elif mode == "g_only":
            if features_t is None or features_s is None:
                features_t, features_s = self.get_features(batch_l, batch_u)
            g_loss = self.adv_c_t(*features_t, mode="g") + self.adv_c_s(*features_s, mode="g")
            outputs = {"g_loss": self._g_lambda * g_loss}
            return outputs


class TriggerEDFilter(nn.Module):
    def __init__(self, nclass:Union[Dict, int], model_name:str, filter_m:int, filter_k:int, hidden_dim=1024, _lambda=10., _g_lambda=0.5,**kwargs):
        super().__init__()
        self.adv_g = SupFilterIE(hidden_dim=hidden_dim, nclass=nclass, filter_m=filter_m, filter_k=filter_k)
        self.adv_c_t = Adv(hidden_dim, _lambda)
        self.adv_c_s = Adv(hidden_dim, _lambda)
        self._g_lambda = _g_lambda

    def clip_params(self,):
        self.adv_c_s.clip_params()
        self.adv_c_t.clip_params()

    def get_d_params(self,):
        return list(self.adv_c_s.parameters()) + list(self.adv_c_t.parameters())

    def get_g_params(self,):
        return list(self.adv_g.parameters())

    def get_features(self, batch_l, batch_u):
        lm_inputs_l = {"input_ids": batch_l["input_ids"], "attention_mask": batch_l["attention_mask"]}
        encoded_langs_l = [encoded if self.adv_g.dim_map is None else self.adv_g.dim_map(encoded) for encoded in self.adv_g.pretrained_lm(lm_inputs_l)]

        lm_inputs_u = {"input_ids": batch_u["input_ids"], "attention_mask": batch_u["attention_mask"]}
        encoded_langs_u = [encoded if self.adv_g.dim_map is None else self.adv_g.dim_map(encoded) for encoded in self.adv_g.pretrained_lm(lm_inputs_u)]

        n_l = encoded_langs_l[0]; t_l = encoded_langs_l[1]
        n_u = encoded_langs_u[1]; t_u = encoded_langs_u[0]

        nl_mask = batch_l["attention_mask"][0]; tl_mask = batch_l["attention_mask"][1]
        nu_mask = batch_u["attention_mask"][1]; tu_mask = batch_u["attention_mask"][0]
        return (t_l, n_u), (t_u, n_l), (tl_mask, nu_mask), (tu_mask, nl_mask)
        # features_target, features_source

    def _task_step(self, feature_ln, feature_lt, labels, eval_lang=None, self_training=False, return_outputs=False):
        if eval_lang is not None or self_training:
            print(f"wrong {eval_lang}")
            raise RuntimeError
        outputs = self.adv_g.event_forward(encoded=[feature_ln, feature_lt], label=labels, self_training=self_training, loss_lang=eval_lang)
        if return_outputs:
            return outputs
        else:
            return outputs["loss"]

    def d_forward(self, features_t, features_s):
        return self.adv_c_t(*features_t, mode="d") + self.adv_c_s(*features_s, mode="d")

    def g_forward(self, batch_l, batch_u, features_t=None, features_s=None, mask_t=None, mask_s=None, warmup=False, self_training=False):
        if features_t is None or features_s is None:
            features_t, features_s, mask_t, mask_s = self.get_features(batch_l, batch_u)
        eval_lang_l = None if "eval_lang" not in batch_l else batch_l["eval_lang"]
        outputs = self._task_step(
            feature_ln=features_s[1],
            feature_lt=features_t[0],
            labels=batch_l["trigger_labels"],
            eval_lang=eval_lang_l,
            return_outputs=True)
        if self_training:
            outputs["loss"] += 0.1 * self._task_step(
                feature_ln=features_t[1],
                feature_lt=features_s[0],
                labels=None,
                eval_lang=None,
                self_training=True,
                return_outputs=False)
        elif "pseudo_trigger_labels" in batch_u:
            outputs["loss"] += self._task_step(
                feature_ln=features_t[1],
                feature_lt=features_s[0],
                labels=batch_u["pseudo_trigger_labels"],
                eval_lang=None,
                return_outputs=False)
            outputs["loss"] *= 0.5

        if not warmup:
            masked_features_t = [ts[ts_mask>0] for ts, ts_mask in zip(features_t, mask_t)]
            masked_features_s = [ts[ts_mask>0] for ts, ts_mask in zip(features_s, mask_s)]
            g_loss = self.adv_c_t(*masked_features_t, mode="g") + self.adv_c_s(*masked_features_s, mode="g")
            outputs["g_loss"] = self._g_lambda * g_loss
        return outputs

    def compute_cross_entropy(self, logits, labels, crit=None):
        return self.adv_g.compute_cross_entropy(logits, labels, crit=crit)

    def forward(self, batch_l, batch_u=None, mode="g", warmup=False):
        '''
            mode = ['g', 'd', 'eval']
        '''
        if mode == 'eval' or batch_u is None:
            return self.adv_g.event_forward(batch=batch_l,loss_lang=1)
        else:
            features_t, features_s = self.get_features(batch_l, batch_u)
            if mode == "g":
                eval_lang_l = None if "eval_lang" not in batch_l else batch_l["eval_lang"]
                outputs = self._task_step(
                    features_ln=features_s[1],
                    features_lt=features_t[0],
                    labels=batch_l["sentence_labels"],
                    copy_labels=True,
                    eval_lang=eval_lang_l,
                    return_outputs=True)
                if "pseudo_sentence_labels" in batch_u:
                    outputs["loss"] += self._task_step(
                        features_ln=features_t[1],
                        features_lt=features_s[0],
                        labels=batch_u["pseudo_sentence_labels"],
                        copy_labels=False,
                        eval_lang=None,
                        return_outputs=False)
                if not warmup:
                    g_loss = self.adv_c_t(*features_t, mode="g") + self.adv_c_s(*features_s, mode="g")
                    outputs["loss"] += self._g_lambda * g_loss
                return outputs
            elif mode == "d":
                return self.adv_c_t(*features_t, mode="d") + self.adv_c_s(*features_s, mode="d")

class RelationEDFilter(nn.Module):
    def __init__(self, nclass:Union[Dict, int], model_name:str, filter_m:int, filter_k:int, hidden_dim=1024, _lambda=10., _g_lambda=1,**kwargs):
        super().__init__()
        self.adv_g = SupFilterIE(hidden_dim=hidden_dim, nclass=nclass, filter_m=filter_m, filter_k=filter_k)
        self.adv_c_t = Adv(hidden_dim, _lambda)
        self.adv_c_s = Adv(hidden_dim, _lambda)
        self._g_lambda = _g_lambda

    def clip_params(self,):
        self.adv_c_s.clip_params()
        self.adv_c_t.clip_params()

    def get_d_params(self,):
        return list(self.adv_c_s.parameters()) + list(self.adv_c_t.parameters())

    def get_g_params(self,):
        return list(self.adv_g.parameters())

    def get_features(self, batch_l, batch_u):
        lm_inputs_l = {"input_ids": batch_l["input_ids"], "attention_mask": batch_l["attention_mask"]}
        encoded_langs_l = [encoded if self.adv_g.dim_map is None else self.adv_g.dim_map(encoded) for encoded in self.adv_g.pretrained_lm(lm_inputs_l)]
        encoded_langs_l = [_span_representations(_encoded, _span) for _encoded, _span in zip(encoded_langs_l, batch_l["entity_spans"])]

        lm_inputs_u = {"input_ids": batch_u["input_ids"], "attention_mask": batch_u["attention_mask"]}
        encoded_langs_u = [encoded if self.adv_g.dim_map is None else self.adv_g.dim_map(encoded) for encoded in self.adv_g.pretrained_lm(lm_inputs_u)]
        encoded_langs_u = [_span_representations(_encoded, _span) for _encoded, _span in zip(encoded_langs_u, batch_u["entity_spans"])]

        n_l = encoded_langs_l[0]; t_l = encoded_langs_l[1]
        n_u = encoded_langs_u[1]; t_u = encoded_langs_u[0]

        nl_mask = torch.sum(batch_l["entity_spans"][0], dim=-1) > 0; tl_mask = torch.sum(batch_l["entity_spans"][1], dim=-1) > 0
        nu_mask = torch.sum(batch_u["entity_spans"][1], dim=-1) > 0; tu_mask = torch.sum(batch_u["entity_spans"][0], dim=-1) > 0
        return (t_l, n_u), (t_u, n_l), (tl_mask, nu_mask), (tu_mask, nl_mask)
        # features_target, features_source

    def _task_step(self, feature_ln, feature_lt, labels, eval_lang=None, self_training=False, return_outputs=False):
        if eval_lang is not None or self_training:
            print(f"wrong {eval_lang}")
            raise RuntimeError
        outputs = self.adv_g.relation_forward(encoded=[feature_ln, feature_lt], label=labels, self_training=self_training, loss_lang=eval_lang)
        if return_outputs:
            return outputs
        else:
            return outputs["loss"]

    def d_forward(self, features_t, features_s):
        return self.adv_c_t(*features_t, mode="d") + self.adv_c_s(*features_s, mode="d")

    def g_forward(self, batch_l, batch_u, features_t=None, features_s=None, mask_t=None, mask_s=None, warmup=False, self_training=False):
        if features_t is None or features_s is None:
            features_t, features_s, mask_t, mask_s = self.get_features(batch_l, batch_u)
        eval_lang_l = None if "eval_lang" not in batch_l else batch_l["eval_lang"]
        outputs = self._task_step(
            feature_ln=features_s[1],
            feature_lt=features_t[0],
            labels=batch_l["relation_labels"],
            eval_lang=eval_lang_l,
            return_outputs=True)
        if self_training:
            outputs["loss"] += 0.1 * self._task_step(
                feature_ln=features_t[1],
                feature_lt=features_s[0],
                labels=None,
                eval_lang=None,
                self_training=True,
                return_outputs=False)
        elif "pseudo_relation_labels" in batch_u:
            outputs["loss"] += self._task_step(
                feature_ln=features_t[1],
                feature_lt=features_s[0],
                labels=batch_u["pseudo_relation_labels"],
                eval_lang=None,
                return_outputs=False)

        if not warmup:
            masked_features_t = [ts[ts_mask>0] for ts, ts_mask in zip(features_t, mask_t)]
            masked_features_s = [ts[ts_mask>0] for ts, ts_mask in zip(features_s, mask_s)]
            g_loss = self.adv_c_t(*masked_features_t, mode="g") + self.adv_c_s(*masked_features_s, mode="g")
            outputs["g_loss"] = self._g_lambda * g_loss
        return outputs

    def compute_cross_entropy(self, logits, labels, crit=None):
        return self.adv_g.compute_cross_entropy(logits, labels, crit=crit)

    def forward(self, batch_l, batch_u=None, mode="g", warmup=False):
        '''
            mode = ['g', 'd', 'eval']
        '''
        if mode == 'eval' or batch_u is None:
            return self.adv_g.relation_forward(batch=batch_l,loss_lang=1)
        else:
            features_t, features_s = self.get_features(batch_l, batch_u)
            if mode == "g":
                eval_lang_l = None if "eval_lang" not in batch_l else batch_l["eval_lang"]
                outputs = self._task_step(
                    features_ln=features_s[1],
                    features_lt=features_t[0],
                    labels=batch_l["relation_labels"],
                    copy_labels=True,
                    eval_lang=eval_lang_l,
                    return_outputs=True)
                if "pseudo_relation_labels" in batch_u:
                    outputs["loss"] += self._task_step(
                        features_ln=features_t[1],
                        features_lt=features_s[0],
                        labels=batch_u["pseudo_relation_labels"],
                        copy_labels=False,
                        eval_lang=None,
                        return_outputs=False)
                if not warmup:
                    g_loss = self.adv_c_t(*features_t, mode="g") + self.adv_c_s(*features_s, mode="g")
                    outputs["loss"] += self._g_lambda * g_loss
                return outputs
            elif mode == "d":
                return self.adv_c_t(*features_t, mode="d") + self.adv_c_s(*features_s, mode="d")
