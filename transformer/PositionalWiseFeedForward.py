import torch.nn as nn
import torch.nn.functional as F


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        # 连续用两个1维卷积，输入输出都是512和2048
        self.w1 = nn.Conv1d(in_channels=model_dim, out_channels=2048, kernel_size=1)
        self.w2 = nn.Conv1d(in_channels=model_dim, out_channels=2048, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        # 标准归一化
        self.layer_norm = nn.LayerNorm(normalized_shape=model_dim)

    def forward(self, x):
        # 维度1和维度2调位置
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # 按照论文里，加入残差
        output = self.layer_norm(x + output)
        return output
