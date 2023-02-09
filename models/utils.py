import torch
import torch.nn as nn
from typing import Union
import math
class SimpleCRFHead(nn.Module):
    def __init__(self, nstate:int=3, trainable:bool=True):
        super().__init__()
        self.tran = nn.Parameter(torch.zeros(nstate, nstate), requires_grad=trainable)
        self.init = nn.Parameter(torch.zeros(nstate), requires_grad=trainable)
        self.mask = nn.Parameter(torch.zeros(nstate, nstate), requires_grad=False)
        self.imask = nn.Parameter(torch.zeros(nstate), requires_grad=False)
        for i in range(nstate):
            if i > 0 and i % 2 == 0:
                self.imask[i] = float("-inf")
            for j in range(nstate):
                if i == 0 :
                    if j > 0 and j % 2 == 0: # O I 
                        self.mask[i, j] = float("-inf")
                elif i % 2 == 1: # B-a I-b
                    if j > 0 and j % 2 == 0 and j != i + 1:
                        self.mask[i, j] = float("-inf")
                elif i % 2 == 0: # I-a I-b
                    if j > 0 and j % 2 == 0 and j != i:
                        self.mask[i, j] = float("-inf")
        self.nstate = nstate
            
    def forward(self, inputs:torch.FloatTensor, path:torch.LongTensor, seq_mask:Union[torch.BoolTensor,       None]=None):
        '''
        inputs: * x L x nstate
        path: * x L
        seq_mask: * x L

        
        '''
        sizes = path.size()
        path_scores = torch.gather(inputs, -1, path.unsqueeze(-1)).view(-1, sizes[-1])
        tran_scores = self.tran[path[..., :-1], path[..., 1:]].view(-1, sizes[-1]-1)
        init_scores = self.init[path[..., 0]].view(-1)
        if seq_mask is None:
            path_scores = torch.sum(path_scores, dim=-1) + init_scores + torch.sum(tran_scores, dim=-1)
        else:
            path_masks = seq_mask.view(-1, sizes[-1]).float()
            tran_masks = torch.logical_and(seq_mask[..., :-1], seq_mask[..., 1:]).view(-1, sizes[-1]-1).      float()
            path_scores = torch.sum(path_scores * path_masks, dim=-1) + init_scores + torch.sum(tran_scores*  tran_masks, dim=-1)
            seq_mask[..., :-1] = torch.logical_and(seq_mask[..., :-1], ~seq_mask[..., 1:])
        
        path_scores = path_scores.view(sizes[:-1])
        
        previous = torch.zeros_like(inputs[..., 0, :]).unsqueeze(-1) + (self.init + self.imask).view(*([1]*   (len(sizes)-1)+[self.nstate, 1]))
        previous = previous + inputs[..., 0, :].unsqueeze(-1)
        tran = (self.tran + self.mask).view(*([1]*(len(sizes)-1)+[self.nstate, self.nstate]))
        scores = torch.zeros_like(inputs[..., 0, 0]).detach()
        for step in range(1, sizes[-1]): 
            previous = previous + tran + inputs[..., step, :].unsqueeze(-2)
            previous = torch.logsumexp(previous, dim=-2)
            if seq_mask is not None and torch.any(seq_mask[..., step]):
                scores[seq_mask[..., step]] = torch.logsumexp(previous[seq_mask[..., step]], dim=-1)
                
            previous = previous.unsqueeze(-1)
        if seq_mask is None:
            scores = torch.logsumexp(previous, dim=[-2,-1])

        return scores - path_scores
    def prediction(self, inputs:torch.FloatTensor):
        '''
        inputs: * x L x nstate
        '''
        states = inputs[..., 0, :]
        tran = (self.tran+self.mask).view(*([1]*(len(inputs.size())-2)+[self.nstate, self.nstate]))
        init = (self.init+self.imask).view(*([1]*(len(inputs.size())-2)+[self.nstate]))
        states = states + init
        path = torch.zeros_like(inputs).long()
        
        for step in range(1, inputs.size(-2)):
            next_states = inputs[..., step, :].unsqueeze(-2) + tran + states.unsqueeze(-1)
            states, index = torch.max(next_states, dim=-2)
            
            if step > 1:
                path = torch.gather(path, -1, index.repeat_interleave(inputs.size(-2), -1).view(*inputs.      size()[:-2], inputs.size(-1), inputs.size(-2)).transpose(-2, -1))
            path[..., step-1, :] = index
        
        score, path_index = torch.max(states, dim=-1)
        pred = torch.gather(path, -1, path_index.unsqueeze(-1).repeat_interleave(inputs.size(-2), -1).        unsqueeze(-1))
        pred = pred.squeeze(-1)
        pred[..., -1] = path_index
        return pred, score


class Bilinear(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        _weight = torch.randn(out_features, in1_features, in2_features) * math.sqrt(2 / (in1_features + in2_features))
        self.weight = nn.Parameter(_weight)
        if bias:
            _bias = torch.ones(out_features) * math.sqrt(2 / (in1_features + in2_features))
            self.bias = nn.Parameter(_bias)
        else:
            self.bias = None
        self.out_features = out_features
        self.in1_features = in1_features
        self.in2_features = in2_features

    def forward(self, input1, input2):
        # B x n x d
        assert len(input1.size()) == len(input2.size())
        input_dims = len(input1.size())
        weight_size = [1] * (input_dims-2) + list(self.weight.size())
        bias_size = [1] * (input_dims-2) + [self.out_features] + [1, 1]
        weight = self.weight.view(*weight_size)
        if self.bias is not None:
            bias = self.bias.view(*bias_size)
        input1 = input1.unsqueeze(-3)
        input2 = input2.unsqueeze(-3).transpose(-2, -1)
        outputs = bias + torch.matmul(input1,
                                     torch.matmul(self.weight.unsqueeze(0),
                                                  input2))
        return outputs.permute(*list(range(0, input_dims-2)), input_dims-1, input_dims, input_dims-2)
    

class BiClassifier(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_linear = nn.Linear(in1_features, 1024, bias=bias)
        self.in2_linear = nn.Linear(in2_features, 1024, bias=False)
        self.out_linear = nn.Linear(1024, out_features, bias=bias)
    def forward(self, input1, input2):
        in1 = self.in1_linear(input1).unsqueeze(-2)
        in2 = self.in2_linear(input2).unsqueeze(-3)
        return self.out_linear(torch.relu(in1 + in2))