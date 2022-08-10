import torch.nn as nn
from Attention.transformer.Attention import MultiHeadAttention
from Attention.transformer.PositionalWiseFeedForward import PositionalWiseFeedForward
from Attention.transformer.PositionEncoding import PositionalEncoding
from Attention.transformer.util import padding_mask


class EncoderLayer(nn.Module):
    # Encoder中的一层，论文中写的是六层叠加为Encoder
    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.0):
        """
        流程： Multi-head Attention -> feedForward
        :param model_dim: 模型的输入维度512
        :param num_heads: 多头注意力的头数
        :param ffn_dim: Position-wise Feed-Forward network中一维卷积的输出维度
        :param dropout: dropout比例
        """
        super(EncoderLayer, self).__init__()
        # 用多头注意力
        self.attention = MultiHeadAttention(model_dim=model_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        # 使用定义的Position-wise Feed-Forward network
        self.feed_forward = PositionalWiseFeedForward(model_dim=model_dim,
                                                      ffn_dim=ffn_dim,
                                                      dropout=dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(key=inputs, value=inputs,
                                            query=inputs, attn_mask=attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    # 多层EncoderLayer组成Encoder
    def __init__(self, vocab_size, max_seq_len, num_layers=6,
                 model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        """
        :param vocab_size:
        :param max_seq_len:
        :param num_layers: Encoder由多少层layer叠加而成，论文中给出：6
        :param model_dim: 默认：512，论文中给出
        :param num_heads: 几头注意力，默认：8
        :param ffn_dim: Position-wise Feed-Forward network中一维卷积的输出维度
        :param dropout: dropout比例
        """
        super(Encoder, self).__init__()
        # 叠加六层encoder_layer
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        # 加一是因为填充了padding
        # padding_idx=0，即第一行为填充的padding
        self.seq_embedding = nn.Embedding(num_embeddings=vocab_size + 1,
                                          embedding_dim=model_dim,
                                          padding_idx=0)
        # 定义位置编码
        self.pos_embedding = PositionalEncoding(d_model=model_dim,
                                                max_seq_len=max_seq_len)

    def forward(self, inputs, inputs_len):
        """
        :param inputs:
        :param inputs_len:
        :return:
        """
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            # 输出output以及attention
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions
