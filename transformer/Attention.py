# 针对Attention is all u need 论文中的ScaledDotProductAttention的实现
import torch.nn as nn
import torch
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scaled=None, attention_mask=None):
        """
        整个流程就是:
        Q dot K^T -> Scale -> Mask -> softmax -> attention dot V
        :param q: query张量， shape为[B, L, D] , L 为长度， D为维度
        :param k: key张量， shape为[B, L, D]
        :param v: Value张量, shape为[B, L, D]
        :param scaled: 缩放因子，一个浮点标量
        :param attention_mask: mask张量， 形状为 [B, L_q, D_k]
        :return: context张量和attention张量
        """
        # 计算q dot K^T
        # 算出来的张量结构应该是[B, l_q, d_k]
        attention = torch.bmm(q, k.transpose(1, 2))
        if scaled:
            attention = attention * scaled
        if attention_mask:
            # 给需要进行mask的地方设置值为负无穷
            attention = attention.masked_fill_(attention_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 使用dropout
        attention = self.dropout(attention)
        # attention dot v 得到上下文信息,这里是从论文中得到的
        context = torch.bmm(attention, v)
        return context, attention


# 针对Attention is all u need 论文中的MultiHeadAttention的实现
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        """
        :param model_dim: 模型输入向量维度，默认512(论文中定义)
        :param num_heads: 模型的注意力头数量，默认8(超参数，论文中定义)
        :param dropout: dropout算法丢弃比
        """
        super(MultiHeadAttention, self).__init__()
        # 论文中得出，64
        self.dim_per_head = model_dim // num_heads
        # 8头
        self.num_heads = num_heads
        # 使用简单线性层算出w_q, w_k, w_v
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        # 调用上面封装的scaledDotProductAttention
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        # 定义dropout算法
        self.dropout = nn.Dropout(dropout)
        # layer norm 层，根据论文中的定义
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attention_mask=None):
        # 残差
        residual = query
        # 每个头的维度大小，64
        dim_per_head = self.dim_per_head
        # 多少个头，8
        num_heads = self.num_heads
        # 批度大小
        batch_size = key.size(0)
        # 投影
        # Q = x * w^q
        query = self.linear_q(query)
        # K = x * w^k
        key = self.linear_k(key)
        # V = x * w^v
        value = self.linear_v(value)
        # 按照head数进行分割
        query = query.view(batch_size * num_heads, -1, dim_per_head)  # [bs * num_heads, seq_len ,dim_per_head]
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)

        if attention_mask:
            # 在batch这个维度进行叠加
            attention_mask = attention_mask.repeat(num_heads, 1, 1)
        # 论文中的根号d_k，即8
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attention_mask
        )
        # 前馈层
        output = self.linear_final(context)
        # dropout算法
        output = self.dropout(output)
        # 残差连接后接layer_norm
        output = self.layer_norm(residual + output)

        return output, attention
