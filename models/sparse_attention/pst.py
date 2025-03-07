# ------------------------------------------------------------------------
# Pyramid Sparse Attention
# Copyright (c) 2025 Tsinghua University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm

from math import ceil
from tqdm import tqdm
import copy

from sparse_attention.compresser import AttentionPool
from sparse_attention.sparse_attention import SparseAttention


# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

@staticmethod
def FeedForward(dim, expansion_factor = 4.):
    dim_hidden = int(dim * expansion_factor)
    return nn.Sequential(
        RMSNorm(dim),
        Linear(dim, dim_hidden),
        nn.GELU(),
        Linear(dim_hidden, dim)
    )

class PyramidSparseEncoder(nn.Module):
    def __init__(
        self,
        num_feature_levels,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.,
        block_size = 4,
        num_selected_blocks = 4
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.layers = []
        for _ in range(num_feature_levels):
            attn = SparseAttention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                block_size=block_size,
                num_selected_blocks=num_selected_blocks,
                compress_mlp = AttentionPool(dim_head, block_size)
            )
            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)
            layer = nn.ModuleList([copy.deepcopy(ModuleList([attn, ff])) for i in range(depth)])
            self.layers.append(layer)
        self.norm = RMSNorm(dim)
    
    

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)
        for _ in tqdm(range(sample_num_times)):
            logits = self.forward(out)
            logits = logits[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample), dim = -1)
        return out[..., prompt_seq_len:]


    def forward(self, tokenslist):
        assert len(tokenslist) == self.num_feature_levels
        outputs = []
        # for each feature level
        for i, layer in enumerate(self.layers):
            tokens = tokenslist[i]
            for attn, ff in layer:
                attn_out = attn(tokens)
                tokens = attn_out + tokens
                tokens = ff(tokens) + tokens
            output = self.norm(tokens)
            outputs.append(output)
        return outputs
