import torch
import torch.nn as nn
import numpy as np


# 实现论文中的positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """
        初始化
        :param d_model: 模型的维度，论文中给出的为512
        :param max_seq_len: 文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 根据公式构造出PositionEncoding矩阵
        # pos / 10000^(2i/d_model)
        position_encoding = np.array([
            [
                pos / np.power(10000, 2.0 * (i // 2) / d_model)
                for i in range(d_model)
            ]
            for pos in range(max_seq_len)
        ])
        # 偶数位置使用sin作为编码函数，奇数位置使用cos作为编码函数
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # 在PE矩阵的第一行，加上一行全是0的向量， 代表填充`PAD`的Positional Encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        padding_row = torch.zeros(([1, d_model]))
        # 拼接
        position_encoding = torch.cat([padding_row, torch.from_numpy(position_encoding)])
        # embedding操作，+1是因为增加了`PAD`这个补全位置的编码
        # Word embedding中如果词典增加`UNK`（unknown words）,也需要+1
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """
        :param input_len: 一个张量，形状为[BATCH_SIZE, 1]。
               每一个张量的值代表这一批文本序列中对应的长度。
        :return:返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [
                list(range(1, length + 1)) + [0] * (max_len - length)
                for length in input_len
            ])
        return self.position_encoding(input_pos)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.autograd import Variable

    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 5000)
    y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
