import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


# 拷贝函数
def clones(module, N):
    """
    :param module: 待拷贝的模块
    :param N: 拷贝的次数
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """
    作为transformer的底层结构
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """

        :param encoder: 编码器
        :param decoder: 解码器
        :param src_embed: 源句子的embedding编码
        :param tgt_embed: 目标句子的embedding编码
        :param generator:
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src:
        :param tgt:
        :param src_mask: 源掩码
        :param tgt_mask: 目标掩码
        :return:
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    定义linear+softmax层
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# 编码器
class Encoder(nn.Module):
    def __init__(self, layer, N=6):
        """
        :param layer: 组成Encoder的layer层，众所周知，encoder是由多个layer堆叠而成的
        :param N: 堆叠的层数，论文中为6层
        """
        super(Encoder, self).__init__()
        # 调用克隆函数生成克隆层
        self.layers = clones(layer, N)
        # 进行标准化
        self.norm = LayerNorm(features=layer.size)

    def forward(self, x, mask):
        """
        x和mask要在每一层堆叠的层过一遍
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    论文中给出：在LayerNorm后会加上一个残差结构
    """

    def __init__(self, size, dropout):
        """
        :param size: x的shape
        :param dropout: dropout比例
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(features=size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
         对具有相同尺寸的任何子层进行残差连接
        :param x:
        :param sublayer:
        :return:
        """
        return x + self.dropout(sublayer(self.norm(x)))


# 编码器层，每层有两个子层
# 一个是多头自注意机制，第二个是简单的位置全连接前馈网络
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 解码器
class Decoder(nn.Module):
    """
    和编码器一样，解码器也是由六层子层堆叠而成
    """

    def __init__(self, layer, N):
        """
        :param layer: 组成Decoder的layer层，众所周知，decoder是由多个layer堆叠而成的
        :param N: 堆叠的层数，论文中为6层

        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(features=layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# 解码器层
class DecoderLayer(nn.Module):
    """
    Decoder 由 self-attn，src-attn 和feedforward组成
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 一个decoder子层由三个部分组成
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    以防止位置关注后续位置。这种掩蔽与输出嵌入偏移一个位置相结合，确保了对位置的预测
    只能依赖于位置小于的已知输出
    # 其实就是乘一个上三角矩阵
    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    subsequent_mask_ = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_)


# 计算注意力
def attention(query, key, value, mask=None, dropout=None):
    """
    scaled dot Product Attention
    流程:
    Q dot k^T -> scaled -> mask -> softmax -> result * V^T
    :param query: 
    :param key: 
    :param value: 
    :param mask: 
    :param dropout: 
    :return: 
    """
    #  query的最后一个维度为64，
    d_k = query.size(-1)
    # q * k^T / 根号d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    # 进行mask操作
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # softmax
    p_attn = F.softmax(scores, dim=-1)
    # dropout算法
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# 计算多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, h=8, d_model=512, dropout=0.1):
        """
        :param h: 头的数量，默认为8
        :param d_model: 模型输入维度，默认512
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        # d_k 要等于 d_model / h
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 每一个头的mask都是一致的
            mask = mask.unqueeze(1)
            # 获得batch数量
        n_batches = query.size(0)

        # 对query， key, value 进行投影，由于按维度切分d_model=512/h=8 = 64=d_k
        # 即从d_model => n, -1, h, d_k
        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 投影后的q，k, v计算attention
        context, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 由于要经过最后一个linear层，所以将8头进行拼接
        # h * d_k = 8 * 64 = 512 = d_model
        context = context.transpose(1, 2).contiquous().view(n_batches, -1, self.h * self.d_k)
        # 返回
        return self.linears[-1](context)


# 全连接前馈网络，输入输出维度固定512，内层维度固定2048
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_f=2048, dropout=0.1):
        """
        :param d_model:
        :param d_f: 内层维度
        :param dropout:
        """
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(in_features=d_model, out_features=d_f)
        self.w2 = nn.Linear(in_features=d_f, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> w1(x) -> relu -> dropout -> w2
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 封装自己的embedding
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        :param d_model: 词向量维度
        :param vocab: 输入词维度
        """
        super(Embeddings, self).__init__()
        # 嵌入词向量
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 和普通的embedding的区别，
        # 这里要除以一个根号d_model
        return self.emb(x) * math.sqrt(self.d_model)


# 实现位置编码
# 在论文中，使用sin和cos函数进行位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model:
        :param dropout:
        :param max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # pos / 10000^(2i / d_model)
        # 5000 * 512
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # 偶数位置使用sin，基数位置使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一个维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# 生成完整模型
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    :param src_vocab:
    :param tgt_vocab:
    :param N: 六层堆叠
    :param d_model: 模型维度，默认512
    :param d_ff: 中间层维度，默认2048
    :param h: 多头注意力头数，默认8
    :param dropout: dropout算法比例，默认0.1
    :return:
    """
    # 拷贝函数
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionWiseFeedForward(d_model)
    model = EncoderDecoder(
        # Encoder，由六层encoder layer堆叠
        encoder=Encoder(layer=EncoderLayer(d_model, self_attn=c(attn),
                                           feed_forward=c(ff), dropout=dropout), N=N),
        # Decoder, 由六层decoder layer堆叠
        decoder=Decoder(layer=DecoderLayer(d_model, self_attn=c(attn), src_attn=c(attn),
                                           feed_forward=c(ff), dropout=dropout), N=N),
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        # Linear + softmax
        generator=Generator(d_model, tgt_vocab)
    )
    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(5, 5))
    # plt.imshow(subsequent_mask(20)[0])
    # plt.show()
    # plt.figure(figsize=(15, 5))
    # pe = PositionalEncoding(20, 0)
    # y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    # plt.show()
