# transformer由decoder和encoder组合而成
import torch.nn as nn
from Attention.transformer.Decoder import Decoder
from Attention.transformer.Encoder import Encoder
from Attention.transformer.util import padding_mask


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        """
        :param src_vocab_size:
        :param src_max_len:
        :param tgt_vocab_size:
        :param num_layers:
        :param model_dim:
        :param num_heads:
        :param ffn_dim:
        :param dropout:
        """
        super(Transformer, self).__init__()
        # 实例化encoder对象
        self.encoder = Encoder(
            vocab_size=src_vocab_size, max_seq_len=src_max_len, num_layers=num_layers,
            model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout
        )
        # 实例化decoder对象
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size, max_seq_len=tgt_max_len, num_layers=num_layers,
            model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout
        )
        # 实例化线性层
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        # 实例化
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        output, enc_self_attn = self.encoder(inputs=src_seq, inputs_len=src_len)

        output, dec_self_attn, ctx_attn = self.decoder(
            tgt_seq, tgt_len, output, context_attn_mask
        )

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn
