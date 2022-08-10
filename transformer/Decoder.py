import torch
import torch.nn as nn
from Attention.transformer.Attention import MultiHeadAttention
from Attention.transformer.PositionalWiseFeedForward import PositionalWiseFeedForward
from Attention.transformer.PositionEncoding import PositionalEncoding
from Attention.transformer.util import padding_mask, sequence_mask


# 定义decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim=2048, dropout=0.0):
        """
        :param model_dim: 模型的输入维度:512
        :param num_heads: 多头注意力的头数
        :param ffn_dim: Position-wise Feed-Forward network中一维卷积的输出维度
        :param dropout: dropout比例
        """
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_input, enc_outputs,
                self_attn_mask=None, context_attn_mask=None):
        """
        :param dec_input:
        :param enc_outputs:
        :param self_attn_mask:
        :param context_attn_mask:
        :return:
        """
        # key ,value ,query 都是decoder的输入
        dec_output, self_attention = self.attention(
            key=dec_input,
            value=dec_input,
            query=dec_input,
            attention_mask=self_attn_mask
        )

        # 上下文context 的注意力
        # dec_output作为query，key和value是encoder的
        dec_output, context_attention = self.attention(
            enc_outputs, enc_outputs, dec_output, context_attn_mask
        )
        # decoder的输出
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


# 定义Decoder
class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        # 叠加decoder_layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout)
             for _ in range(num_layers)]
        )
        # 这里加1是应为有padding_mask
        self.seq_embedding = nn.Embedding(num_embeddings=vocab_size + 1,
                                          embedding_dim=model_dim,
                                          padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model=model_dim,
                                                max_seq_len=max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        """
        :param inputs:
        :param inputs_len:
        :param enc_output:
        :param context_attn_mask:
        :return:
        """
        # 词向量嵌入
        output = self.seq_embedding(inputs)
        # 位置嵌入
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)

        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []
        # 遍历decoder_layer
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask
            )
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions
