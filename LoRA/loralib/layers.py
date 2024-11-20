#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRAModule, self).__init__()
        self.rank = rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        config ,
        r: int = 0,
        if_lora = True,
        shared_lora = None,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha = lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.config = config

        self.head_dim = int(out_features/config.num_attention_heads)
        self.fan_in_fan_out = fan_in_fan_out
        self.if_lora = if_lora
        self.lora_layer_importance_batch = []

        # Actual trainable parameters
        if r > 0 and if_lora:
            self.lora = LoRAModule(in_features, out_features, r)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        elif r > 0 and not if_lora and shared_lora != None:
            self.lora = LoRAModule(in_features, out_features, r)
            self.lora.lora_A = shared_lora.lora_A
            self.lora.lora_B = shared_lora.lora_B
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora.lora_B.weight)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if torch.isnan(self.lora.lora_A.weight).any().item():
            print("1111111111111111111111111111111111111111")

        if self.r > 0:
            result = F.linear(x, T(self.weight), bias=self.bias)
            lora_state = (self.lora(self.lora_dropout(x))) * self.scaling      
            result += lora_state
            if self.config.if_stage2:
                hidden_shape = lora_state.shape
                importance_matrix = lora_state.view(hidden_shape[0],hidden_shape[1],1,hidden_shape[2]) @ lora_state.view(hidden_shape[0],hidden_shape[1],hidden_shape[2],1).detach()
                self.lora_layer_importance_batch.append(torch.sum(importance_matrix).item())

            return result