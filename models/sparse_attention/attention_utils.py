# ------------------------------------------------------------------------
# Pyramid Sparse Attention
# Copyright (c) 2025 Tsinghua University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F

# einstein notation
from einops import einsum, rearrange


def max_neg_value(t):
    """Return the maximum negative value for the tensor's dtype."""
    return -torch.finfo(t.dtype).max

def LinearMultiGroupAttention(q, k, v, mask=None, scale=None, eps=1e-6):
    """
    执行线性多组注意力（Linear Multi-Group Attention, MGA）计算。

    该函数将查询张量按需拆分为多个组，基于线性注意力机制计算分组查询、键和值之间的注意力，
    并支持可选的注意力掩码（采用简化处理方式），最终输出注意力结果。

    参数：
        q (Tensor)：查询张量，形状为 [batch, heads, seq_len, dim_head]。
        k (Tensor)：键张量，形状为 [batch, kv_heads, kv_seq_len, dim_head]。
        v (Tensor)：值张量，形状为 [batch, kv_heads, kv_seq_len, dim_head]。
        mask (Tensor, 可选)：注意力掩码，可以广播到注意力分数的形状。默认为 None。
        return_sim (bool)：是否返回相似度分数。默认为 False。在线性注意力中，相似度不会显式计算，
                          若设为 True，则返回 None。
        scale (float, 可选)：点积注意力的缩放因子。若为 None，则默认为 q.shape[-1]**-0.5。

    返回：
        Tensor：注意力输出，形状为 [batch, heads, seq_len, dim_head]。
                若 return_sim 为 True，则返回 (output, similarity)，其中 similarity 为 None。
    """
    # 1. 如果未提供缩放因子，则设置默认值
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    
    # 2. 获取 q 和 k 的头数
    q_heads, k_heads = q.shape[1], k.shape[1]
    # 计算每个键/值头对应的分组查询数量
    num_grouped_queries = q_heads // k_heads
    
    # 3. 重排查询张量 q 的形状为 [batch, kv_heads, grouped_queries, seq_len, dim_head]
    #    输入 q 的形状为 [b, heads*grouped_queries, seq_len, dim_head]
    q = rearrange(q, 'b (h qh) i d -> b h qh i d', qh=num_grouped_queries)
    
    
    # 5. 应用特征映射 phi(x) = elu(x) + 1，以近似 softmax 行为
    phi_q = (F.elu(q) + 1) * scale  # 形状：[b, h, qh, i, d]
    phi_k = (F.elu(k) + 1) * scale  # 形状：[b, h, j, d]
    
    # 6. 处理掩码（简化方式）
    #    如果提供了掩码，则将其应用到 phi_k 和 v 上，屏蔽掉被掩码的位置
    if mask is not None:
        # 假设掩码形状为 [b, 1, i, j] 或兼容形状，广播到 [b, h, j]
        # 调整掩码以匹配 k 和 v 的维度：[b, h, j, 1]
        mask = mask[..., None]  # 添加广播维度
        # 将掩码应用到 phi_k 和 v（假设掩码与 kv_seq_len 维度对齐）
        phi_k = phi_k * mask  # 形状：[b, h, j, d]
        v = v * mask          # 形状：[b, h, j, d]
    
    # 7. 计算分母中的 sum_phi_k
    sum_phi_k = phi_k.sum(dim=2)  # 形状：[b, h, d]
    
    # 8. 计算线性注意力的分子部分 phi_k^T V
    #    phi_k：[b, h, j, d]
    #    v：    [b, h, j, d]
    #    kv：   [b, h, d, d]
    kv = einsum(phi_k, v, 'b h j d, b h j e -> b h d e')
    
    # 9. 计算分子：phi_q (phi_k^T V)
    #    phi_q：[b, h, qh, i, d]
    #    kv：   [b, h, d, e]
    #    out：  [b, h, qh, i, e]
    out = einsum(phi_q, kv, 'b h qh i d, b h d e -> b h qh i e')
    
    # 10. 计算分母：phi_q (sum_phi_k)
    #     sum_phi_k：[b, h, d]
    #     denom：    [b, h, qh, i]
    denom = einsum(phi_q, sum_phi_k, 'b h qh i d, b h d -> b h qh i')
    
    # 11. 归一化输出
    out = out / (denom[..., None] + eps)  # 形状：[b, h, qh, i, e]
    
    # 12. 重排输出以匹配原始形状：[b, heads, seq_len, dim_head]
    out = rearrange(out, 'b h qh i d -> b (h qh) i d')

    return out

def MultiGroupAttention(q, k, v, mask=None, return_sim=False, scale=None):
    """
    Perform Multi-Group Attention (MGA) computation.

    This function splits the query tensor into multiple groups if necessary,
    computes the scaled dot-product attention between the grouped queries and keys,
    applies an optional mask, and then computes the attention output.
    
    Args:
        q (Tensor): Query tensor of shape [batch, heads, seq_len, dim_head].
        k (Tensor): Key tensor of shape [batch, kv_heads, kv_seq_len, dim_head].
        v (Tensor): Value tensor of shape [batch, kv_heads, kv_seq_len, dim_head].
        mask (Tensor, optional): Attention mask that can be broadcasted to the shape
                                 of the attention scores. Defaults to None.
        return_sim (bool): Whether to return the similarity scores along with the output.
                           Defaults to False.
        scale (float, optional): Scaling factor for dot-product attention. If None,
                                 defaults to q.shape[-1]**-0.5.
    
    Returns:
        Tensor: Attention output of shape [batch, heads, seq_len, dim_head], or
                (output, similarity) if return_sim is True.
    """
    # 1. 若未提供缩放因子，则默认采用 dim_head 的倒数平方根进行缩放
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    
    # 2. 获取 q 与 k 中的头数
    q_heads, k_heads = q.shape[1], k.shape[1]
    # 计算每个 k_head 对应多少个 grouped query
    num_grouped_queries = q_heads // k_heads
    
    # 3. 将查询张量 q 重排为形状 [batch, heads, grouped_queries, seq_len, dim_head]
    #    输入 q 的形状为 [b, heads*grouped_queries, seq_len, dim_head]
    q = rearrange(q, 'b (h qh) i d -> b h qh i d', qh=num_grouped_queries)
    
    # 4. 计算查询与键的点积相似度，并进行缩放
    #    q 的形状为 [b, h, grouped_queries, seq_len, d]
    #    k 的形状为 [b, h, kv_seq_len, d]（注意 kv_heads 与 heads 对齐）
    #    计算后 sim 的形状为 [b, h, grouped_queries, seq_len, kv_seq_len]
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    
    # 5. 获取一个足够小的负数，用于掩码位置的填充（使 softmax 后权重接近 0）
    mask_value = -torch.finfo(sim.dtype).max  # 也可以使用自定义的 max_neg_value 函数
    
    # 6. 如果提供了 mask，则将 mask 为 False 的位置填充为极小值
    if mask is not None:
        sim = sim.masked_fill(~mask, mask_value)
    
    # 7. 对最后一维（键的维度）进行 softmax，得到注意力权重
    attn = sim.softmax(dim=-1)
    
    # 8. 利用注意力权重对值进行加权求和，计算注意力输出
    #    attn 的形状为 [b, h, grouped_queries, seq_len, kv_seq_len]
    #    v 的形状为 [b, h, kv_seq_len, d]
    #    得到 out 的形状为 [b, h, grouped_queries, seq_len, d]
    out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')
    
    # 9. 将头与 grouped query 维度合并，得到形状 [b, heads, seq_len, d] 其中 heads = h * grouped_queries
    out = rearrange(out, 'b h qh i d -> b (h qh) i d')
    
    # 10. 如果需要返回相似度，则对 sim 进行重排后一起返回
    if return_sim:
        sim = rearrange(sim, 'b h qh i j -> b (h qh) i j')
        return out, sim
    return out

def LinearAttention(fq, fk, fv, fmask, seq_len, fine_num_grouped_queries=1, scale=1.0, eps=1e-6):
    """
    Linear Attention 计算，将传统的 softmax 注意力替换为基于核函数的分解方法。
    
    参数:
      fq: 查询张量，形状 [b, h, 1, n, head_d]
      fk: 键张量，形状 [b, h, n, L, head_d]
      fv: 值张量，形状 [b, h, n, L, head_d]
      fmask: 掩码张量，形状 [b, h, 1, n, head_d]，用于屏蔽不需要的 key 位置
      seq_len: 原始序列长度 n
      fine_num_grouped_queries: 分组查询的数量（默认为1）
      scale: 缩放因子（此处未直接使用，可根据需求保留或移除）
      eps: 防止除零的小常数
      
    返回:
      fine_attn_out: 经过线性注意力计算后的输出，形状 [b, (h * fine_num_grouped_queries), seq_len, head_d]
    """
    
    # 1. 将查询重排成 [b, h, qh, n, head_d]
    fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh=fine_num_grouped_queries)  # [b, h, qh, n, head_d]

    # 2. 应用正值核函数（例如 ELU + 1）映射查询和键
    f_q = F.elu(fq) + 1    # [b, h, qh, n, head_d]
    f_k = F.elu(fk) + 1    # [b, h, n, M, head_d]

    # 3. 对键应用掩码：将 mask 中被屏蔽的位置对应 f_k 置零  
    # 将 fmask 从 [b, h, 1, n, M] 重排为 [b, h, n, M, 1]
    fmask = rearrange(fmask, 'b h 1 n M -> b h n M 1')
    #    fmask 的形状为 [b, h, 1, n, M]，通过自动广播，乘到 f_k 的 [b, h, n, M, head_d] 上
    f_k = f_k * fmask.to(f_k.dtype)

    # 4. 对于每个查询位置（序列维度 n），只在对应位置的键上做聚合：
    #    计算聚合的 KV 矩阵和聚合的键和
    #    KV 聚合：对块维度 M 求和时，用键对应值加权求和
    KV = torch.einsum('b h n m d, b h n m e -> b h n d e', f_k, fv)   # [b, h, n, head_d, head_d]
    #    聚合的键和：在每个位置上把 f_k 在块维度 M 求和
    K_sum = f_k.sum(dim=3)  # [b, h, n, head_d]

    # 5. 计算线性注意力输出
    #    对于每个查询位置 i 和每个 grouped query，计算：
    #      output = f_q[i] * (KV[i]) / (f_q[i] 与 K_sum[i] 点积)
    #    其中 KV[i] 的形状为 [b, h, n, head_d, head_d]，f_q[i] 为 [b, h, qh, n, head_d]
    numerator = torch.einsum('b h g n d, b h n d e -> b h g n e', f_q, KV)  # [b, h, qh, n, head_d]
    denom   = torch.einsum('b h g n d, b h n d -> b h g n', f_q, K_sum) + eps   # [b, h, qh, n]

    # 最终每个查询位置的输出为：
    fine_attn_out = numerator / denom.unsqueeze(-1)  # [b, h, qh, n, head_d]

    # 6. 重排输出：将 grouped query 维度和头维度合并，并截取前 seq_len 个位置
    fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')
    fine_attn_out = fine_attn_out[..., :seq_len, :]
    
    return fine_attn_out

def FineAttention(q, fk, fv, fmask, seq_len, scale, fine_num_grouped_queries=1):
    """
    计算精细注意力（Fine Attention），输入包括查询、键、值和掩码，
    并返回经过注意力加权后的输出。
    
    参数：
      q: 查询张量，形状为 [batch, heads * grouped_queries, n, dim_head]
         其中 n 为序列长度，grouped_queries 通常为 1 或更多。
      fk: 键张量，形状为 [batch, heads, n, j, dim_head]
          这里 j 表示每个位置对应的键块内的长度。
      fv: 值张量，形状与 fk 相同：[batch, heads, n, j, dim_head]
      fmask: 掩码张量，布尔类型，形状应与相似度矩阵 fsim 广播兼容，
             例如 [batch, heads, 1, n, j]。在屏蔽的位置应为 False。
      seq_len: 原始序列长度 n，用于截取最终输出的序列部分。
      scale: 缩放因子，通常设置为 (dim_head)**(-0.5)。
      fine_num_grouped_queries: grouped query 的数量，默认为 1。
    
    返回：
      fine_attn_out: 经过精细注意力计算后的输出，
                     形状为 [batch, heads * grouped_queries, seq_len, dim_head]
    """
    # 1. 将查询 q 重排为形状 [batch, heads, grouped_queries, n, dim_head]
    #    输入 q 的形状为 [b, h * qh, n, d]，转换后 q 分解为多个 grouped query
    fq = rearrange(q, 'b (h qh) ... -> b h qh ...', qh=fine_num_grouped_queries)
    
    # 2. 计算查询与键的点积相似度
    #    使用爱因斯坦求和计算点积，结果 fsim 的形状为 [b, h, grouped_queries, n, j]
    fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * scale
    
    # 3. 获取一个极小的负数，用作被屏蔽位置的填充值
    mask_value = max_neg_value(fsim)
    
    # 4. 根据 fmask 对 fsim 进行屏蔽处理
    #    fmask 的形状应能广播至 fsim 的形状，通常 fmask 为 [b, h, 1, n, j]
    fsim = fsim.masked_fill(~fmask, mask_value)
    
    # 5. 对相似度进行 softmax 归一化，得到注意力权重 fattn
    fattn = fsim.softmax(dim=-1)
    
    # 6. 利用注意力权重对值 fv 进行加权求和
    #    输出形状为 [b, h, grouped_queries, n, dim_head]
    fine_attn_out = einsum(fattn, fv, 'b h qh i j, b h i j d -> b h qh i d')
    
    # 7. 将头与 grouped query 维度合并，得到形状 [b, (h * grouped_queries), n, dim_head]
    fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')
    
    # 8. 截取前 seq_len 个序列位置，保证输出与原始序列长度一致
    fine_attn_out = fine_attn_out[..., :seq_len, :]
    
    return fine_attn_out
