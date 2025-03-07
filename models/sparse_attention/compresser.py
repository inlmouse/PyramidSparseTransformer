import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import math
from einops import einsum, rearrange
from einops.layers.torch import EinMix as Mix

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# start accumulating some types of compression networks

class Conv2DCompress(nn.Module):
    def __init__(self, dim_head, stride):
        """
        初始化 Conv2DCompress 模块。
        
        参数:
            dim_head (int): 输入张量的通道维度 (d)。
            stride (int): 卷积的步幅 (s)。
        """
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(dim_head, dim_head, 
                              kernel_size=stride, 
                              stride=stride)

    def forward(self, kv, w, h):
        """
        前向传播，自动处理填充并进行二维卷积压缩。
        
        参数:
            kv (torch.Tensor): 输入张量，形状为 [b, h, n, d]，其中 n = w * h。
            w (int): 输入的空间宽度。
            h (int): 输入的空间高度。
        
        返回:
            torch.Tensor: 压缩后的张量，形状为 [b, h, w_compressed * h_compressed, d]。
        """
        b, h, n, d = kv.shape
        assert n == w * h, "n 必须等于 w * h"

        # 重排为 [b, h, w, h, d]
        kv = rearrange(kv, 'b h (w h) d -> b h w h d', w=w, h=h)
        
        # 重排为 [b * h, d, w, h]，准备进行卷积
        kv = rearrange(kv, 'b h w h d -> (b h) d w h')
        
        # 计算填充量，使 w 和 h 能被步幅整除
        pad_w = (math.ceil(w / self.stride) * self.stride) - w
        pad_h = (math.ceil(h / self.stride) * self.stride) - h
        
        # 如果需要填充，则应用填充
        if pad_w > 0 or pad_h > 0:
            kv = F.pad(kv, (0, pad_h, 0, pad_w), mode='constant', value=0)
        
        # 应用二维卷积
        compressed = self.conv(kv)
        
        # 获取压缩后的宽度和高度
        w_compressed = compressed.shape[2]
        h_compressed = compressed.shape[3]
        
        # 重排回 [b, h, w_compressed * h_compressed, d]
        compressed = rearrange(compressed, '(b h) d w h -> b h (w h) d', b=b, h=h)
        
        return compressed

class ConvLinearCompress(Module):
    """
    used successfully in an old google brain paper, https://github.com/lucidrains/memory-efficient-attention-pytorch
    grouped convolutions so each head get its own parameters
    """

    def __init__(
        self,
        heads,
        dim_head,
        compress_block_size
    ):
        super().__init__()
        self.heads = heads
        self.conv = nn.Conv1d(heads * dim_head, heads * dim_head, compress_block_size, stride = compress_block_size, groups = heads)

    def forward(
        self,
        kv # Float['b h w n d']
    ):

        kv = rearrange(kv, 'b h w n d -> b (h d) (w n)')

        compressed = self.conv(kv)

        return rearrange(compressed, 'b (h d) n -> b h n d', h = self.heads)

# attention pool used by enformer, deepmind's genetic attention network

class AttentionPool(Module):
    def __init__(
        self,
        dim_head,
        compress_block_size
    ):
        super().__init__()
        self.to_attn_logits = nn.Linear(dim_head, dim_head, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim_head))

    def forward(
        self,
        kv
    ):

        attn_logits = self.to_attn_logits(kv)

        attn = attn_logits.softmax(dim = -2)

        compressed = einsum(kv, attn, 'b h w n d, b h w n d -> b h w d')

        return compressed

# mlp per head

class GroupedMLP(Module):
    def __init__(
        self,
        dim_head,
        compress_block_size,
        heads,
        expand_factor = 1.,
    ):
        super().__init__()

        dim = dim_head * compress_block_size
        dim_hidden = int(dim * expand_factor)
        dim_out = dim_head

        self.net = nn.Sequential(
            Mix('b h w i -> b h w o', weight_shape = 'h i o', bias_shape = 'h o', h = heads, i = dim, o = dim_hidden),
            nn.ReLU(),
            Mix('b h w i -> b h w o', weight_shape = 'h i o', bias_shape = 'h o', h = heads, i = dim_hidden, o = dim_out),
        )

    def forward(
        self,
        kv
    ):
        kv = rearrange(kv, 'b h w n d -> b h w (n d)')

        compressed = self.net(kv)

        return compressed
