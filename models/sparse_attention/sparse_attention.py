# ------------------------------------------------------------------------
# Pyramid Sparse Attention
# Copyright (c) 2025 Tsinghua University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from NSA(Native Sparse Attention: Hardware-Aligned and NativelyTrainable Sparse Attention) (https://arxiv.org/pdf/2502.11089)
# ------------------------------------------------------------------------
from __future__ import annotations

from copy import deepcopy
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList
from rotary_embedding_torch import RotaryEmbedding

# einstein notation
import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

# standerd attention modules
from .attention_utils import MultiGroupAttention, FineAttention, LinearAttention

# b - batch
# h - heads
# qh - grouped query heads
# n - sequence (token level or compressed)
# w - windows, for fine or compressed
# i, j - query / key sequence
# d - feature dimension
# s - strategies


# Helper Functions
def exists(v):
    """Check if a value is not None."""
    return v is not None

def default(v, d):
    """Return value if it exists, otherwise return default."""
    return v if exists(v) else d

def round_down_mult(n, mult):
    """Round down `n` to the nearest multiple of `mult`."""
    return n // mult * mult

def round_up_mult(n, mult):
    """Round up `n` to the nearest multiple of `mult`."""
    return ceil(n / mult) * mult

def divisible_by(num, den):
    """Check if `num` is divisible by `den`."""
    return (num % den) == 0

def pad_at_dim(t, pad, dim=-1, value=0.):
    """Pad tensor `t` at specified dimension with given value."""
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)

def straight_through(t, target):
    """Straight-through estimator for gradient computation."""
    return t + (target - t).detach()

class AggregatedGated(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int
        ):
        super().__init__()
        
        aggregated_gated_mlp = nn.Linear(dim, 2 * heads)
        nn.init.zeros_(aggregated_gated_mlp.weight)
        aggregated_gated_mlp.bias.data.copy_(torch.tensor([-2., -2.] * heads))
        self.to_aggregated_gated = nn.Sequential(
            aggregated_gated_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h=heads)
        )

    def forward(self, inp: Tensor, compressed_attn_out: Tensor, fine_attn_out: Tensor) -> Tensor:
        selfgate_weights = self.to_aggregated_gated(inp)
        out = einsum(selfgate_weights, stack([compressed_attn_out, fine_attn_out]), 'b h n s, s b h n d -> b h n d')
        return out



# SparseAttention Class
class SparseAttention(Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        block_size: int,
        num_selected_blocks: int,
        kv_heads: int | None = None,
        num_compressed_mem_kv: int = 1,
        norm: bool = True,
        use_diff_topk: bool = False,
        query_heads_share_selected_kv: bool = True,
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor: float = 1.,
        aggregated_gated_mlp: Module | None = None
    ):
        """
        Initialize the SparseAttention module, implementing an efficient attention mechanism.

        Args:
            dim (int): Input dimension.
            dim_head (int): Dimension per attention head.
            heads (int): Number of query heads.
            block_size (int): Size of each block for sparse attention.
            num_selected_blocks (int): Number of blocks to select for fine attention.
            kv_heads (int, optional): Number of key/value heads (for GQA). Defaults to heads.
            num_compressed_mem_kv (int): Number of memory compressed key/value pairs.
            norm (bool): Whether to apply RMS normalization. Defaults to True.
            use_diff_topk (bool): Use differentiable top-k selection. Defaults to False.
            query_heads_share_selected_kv (bool): Whether query heads share selected key/values.
            compress_mlp (Module, optional): MLP for compressing key/values. Defaults to None.
            compress_mlp_expand_factor (float): Expansion factor for compress MLP hidden dim.
            aggregated_gated_mlp (Module, optional): MLP for combining strategies.
        """
        super().__init__()

        # Handle Grouped Query Attention (GQA)
        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads), "kv_heads must be <= heads and heads must be divisible by kv_heads"
        assert dim_head * heads == dim, "dim_head * heads must be equal dim"
        self.heads = heads
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads
        self.scale = dim_head ** -0.5

        # Dimensions
        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads
        self.qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)

        # Layers
        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.to_qkv = nn.Linear(dim, sum(self.qkv_split), bias=False)

        # Block configuration
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n=block_size)

        # Compressed memory key/values
        assert num_compressed_mem_kv > 0, "num_compressed_mem_kv must be positive"
        self.num_mem_compress_kv = num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, block_size, dim_head))

        if not exists(compress_mlp):
            compress_dim = block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)

            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # Selection parameters
        self.use_diff_topk = use_diff_topk
        self.query_heads_share_selected_kv = query_heads_share_selected_kv

        # they combine the three sparse branches through a learned combine with sigmoid activation

        # Selfgate
        if not exists(aggregated_gated_mlp):
            self.aggregated_gated_mlp = AggregatedGated(dim=dim, dim_head=dim_head, heads=heads)
        else:
            self.aggregated_gated_mlp = aggregated_gated_mlp

        # Head splitting and merging
        self.split_heads = Rearrange('b n (h d) -> b h n d', d=dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim, bias=False)


    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass of SparseAttention, combining coarse and fine attention mechanisms.

        Args:
            inp (Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        batch, seq_len = inp.shape[:2]
        heads = self.heads

        # Compute block numbers
        compress_divisible_seq_len = round_down_mult(seq_len, self.block_size)
        num_compress_blocks = compress_divisible_seq_len // self.block_size
        fine_divisible_seq_len = round_up_mult(seq_len, self.block_size)
        num_fine_blocks = fine_divisible_seq_len // self.block_size

        # Normalize input
        inp = self.norm(inp)

        # Project to queries, keys, values
        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v)) #[b,h,n,d/h]
        
        # Prepare compressed key/values
        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r=num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r=num_compress_blocks)
        k_compress_input = self.split_compress_window(k[..., :compress_divisible_seq_len, :] + k_pos)
        v_compress_input = self.split_compress_window(v[..., :compress_divisible_seq_len, :] + v_pos)
        # Variables prepended with `c` stands for compressed
        cq = q
        ck = self.k_compress(k_compress_input) # Equation (7) of the Native Sparse Attention paper
        cv = self.v_compress(v_compress_input) #[b,h,n/blocksize, d/h]


        # 1. coarse attention over compressed
        # Incorporate memory compressed key/values
        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = batch)
        num_mem_compress_kv = mem_ck.shape[-2]
        ck = cat((mem_ck, ck), dim = -2)
        cv = cat((mem_cv, cv), dim = -2)

        # Coarse attention over compressed key/values
        compressed_attn_out, csim = MultiGroupAttention(cq, ck, cv, mask = None, return_sim = True)
        
        # for 2. , will give them relative positions with rotary - compressed needs to be handled separately (even if they already have intra block absolute positions)
        # Apply rotary embeddings for fine attention
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)


        # 2. fine attention over selected based on compressed attention logits - variables prepended with `f` stands for the fine attention pathway
        # Select blocks for fine attention based on coarse attention similarities
        importance_scores = csim[..., num_mem_compress_kv:] #[b,h,n,n/blocksize]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)

        # maybe average the compressed attention across each grouped queries (per key / values)

        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)

            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1

        
        # Softmax over importance scores for block selection
        importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
        importance_scores = importance_scores.softmax(dim = -1)
        importance_scores = importance_scores[..., 1:]

        # Select top-k blocks
        selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)
        #selected_importance_values shape [b,h,n,num_selected]
        
    
        # Prepare fine attention inputs
        fq, fk, fv = q, k, v
        fmask = selected_importance_values > 1e-10

        gates = None
        if self.use_diff_topk:
            gates = straight_through(selected_importance_values, 1.)

        else:
            # Pad if sequence length is not block-divisible
            if seq_len < fine_divisible_seq_len:
                remainder = fine_divisible_seq_len - seq_len
                fk = pad_at_dim(fk, (0, remainder), value = 0., dim = -2)
                fv = pad_at_dim(fv, (0, remainder), value = 0., dim = -2)
                fq = pad_at_dim(fq, (0, remainder), value = 0., dim = -2)
                fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)
                selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

            # Rearrange for block selection, select out the spatial crops of keys / values for fine attention
            fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)#[b,h,n/blocksize,blocksize,dim_head]
            fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

            # Expand and gather selected blocks
            if self.query_heads_share_selected_kv:
                fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])#[b,h,n,n/blocksize,blocksize,dim_head]
                fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
            else:
                fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

            selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])#[b,h,n,num_selected,blocksize,dim_head]
            fk = fk.gather(3, selected_block_indices)#[b,h,n,num_selected,blocksize,dim_head]
            fv = fv.gather(3, selected_block_indices)

            
            # differential topk gating
            if self.use_diff_topk:
                fk = einx.multiply('b h i sel, b h i sel j d -> b h i sel j d', gates, fk)

            # Merge selected key/values
            fk, fv = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))#[b,h,n,num_selected*blocksize,dim_head]
            fmask = repeat(fmask, 'b h i w -> b h 1 i (w j)', j = self.block_size)
        
            # Fine attention computation
            #fine_attn_out = LinearAttention(fq,fk,fv,fmask,seq_len,fine_num_grouped_queries)#Even slower?!
            fine_attn_out = FineAttention(fq,fk,fv,fmask,seq_len,fine_num_grouped_queries)

        # Gated combine coarse and fine attention outputs
        out = self.aggregated_gated_mlp(inp, compressed_attn_out, fine_attn_out)

        # Merge heads and project output
        out = self.merge_heads(out)
        out = self.combine_heads(out)
        return out

