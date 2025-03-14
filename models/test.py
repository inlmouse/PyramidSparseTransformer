import torch
from torch import nn
from einops.layers.torch import Rearrange

from sparse_attention import SparseAttention
from sparse_attention.pst import PyramidSparseEncoder, PyramidSparseDecoder

import time

def test_compress_networks():
    from sparse_attention.compresser import AttentionPool

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        block_size = 4,
        num_selected_blocks = 2,
        compress_mlp = AttentionPool(64, 4)
    )

    tokens = torch.randn(2, 33, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape


def test_transformer():
    trans = PyramidSparseEncoder(
        num_feature_levels = 4,
        dim = 256,
        depth = 2,
        dim_head = 32,
        heads = 8,
        block_size = 25,#5*5~7*7 patchify
        num_selected_blocks = 8
    ).to("cpu").to(torch.float32)
    decoder = PyramidSparseDecoder(num_feature_levels=4, dim=256, depth=6)
    tokenslist = [
        torch.randn(4, 76*106, 256).to("cpu").to(torch.float32),
        torch.randn(4, 38*53, 256).to("cpu").to(torch.float32),
        torch.randn(4, 19*27, 256).to("cpu").to(torch.float32),
        torch.randn(4, 10*14, 256).to("cpu").to(torch.float32)
    ]
    #tokens = torch.randn(2, 33, 512)
    decoder(tokenslist)
    output = trans(tokenslist)
    T1 = time.time()
    for i in range(10):
        output = trans(tokenslist)
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000/10))
    #assert tokenslist.shape == output.shape

test_transformer()