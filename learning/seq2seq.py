import torch
import torch.nn as nn
import torch.nn.functional as F

""" 以离散符号的分类任务为例，实现基于注意力机制的seq2seq模型 """


class Seq2seqEncoder(nn.Module):
    """ 实现基于LSTM的编码器 """

    def __init__(self, embedding_dim, hidden_size, src_vocab_size):
        super(Seq2seqEncoder, self).__init__()

        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.embedding_table = nn.Embedding(num_embeddings=src_vocab_size,
                                            embedding_dim=embedding_dim
                                            )

    def forward(self, input_ids):
        input_seq = self.embedding_table(input_ids)
        output_states, (final_h, final_c) = self.lstm_layer(input_seq)

        return output_states, final_h
