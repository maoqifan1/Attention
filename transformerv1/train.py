import math, copy, time
from Attention.transformerv1.model import subsequent_mask, make_model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


# 用于训练源句子和目标句子以及构建掩码
class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建mask来隐藏padding和当前词后面的词(防止作弊)
        :param tgt:
        :param pad:
        :return:
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    """
    通用的训练和评分函数来跟踪损失。传入一个通用的损失计算函数，它也处理参数更新。
    :param data_iter:
    :param model:
    :param loss_compute:
    :return:
    """
    start = time.time()
    total_tokens = 0
    # 统计loss
    total_loss = 0
    tokens = 0
    for idx, batch in enumerate(data_iter):
        # 送入模型
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        # 计算损失
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        # 每50次迭代输出一次信息
        if idx % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (idx, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    """
    优化策略
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        更新参数
        :return:
        """
        self._step += 1
        rate = self.rate()
        # 预设学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        # 记录学习率
        self._rate = rate
        # 更新参数
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
            # 这里调用一个计算学习率的公式
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model_size=model.src_embed[0].d_model, factor=2,
                   warmup=4000,
                   optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    """
    label smoothing 算法

    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# ------------简单的复制任务，给定来自小词汇表的一组随机输入符号，目标是生成相同的符号。------------#
def data_gen(V, batch, nbatches):
    """随机生成数据"""
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """
    计算损失函数
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)  # linear + softmax
        # 损失函数
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # 反向传播
        loss.backward()
        if self.opt is not None:
            self.opt.step()  # 更新参数
            self.opt.optimizer.zero_grad()  # 清空梯度，防止叠加
        return loss.data.item() * norm


# 贪心解码
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(src_vocab=V, tgt_vocab=V, N=2)  # 初始化模型
model_opt = NoamOpt(model_size=model.src_embed[0].d_model, factor=1,
                    warmup=400, optimizer=torch.optim.Adam(model.parameters(),
                                                           lr=0,
                                                           betas=(0.9, 0.98),
                                                           eps=1e-9))
for epoch in range(10):
    model.train()  # 启用dropout
    run_epoch(data_iter=data_gen(V, 30, 20), model=model,
              loss_compute=SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))
