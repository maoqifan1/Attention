import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# # 分块方法得到embedding
# pe1 = image2emb_naive(image, patch_size, weight)
# # oc * ic * kh * kw
# kernel = weight.transpose(0, 1).reshape(-1, args.inp, args.patch_size, args.patch_size)
# # 二维卷积得到embedding
# pe2 = image2emb_conv(image, kernel, patch_size)

# 第一步：将图片转换为embedding 向量序列
def image2emb_naive(image, patch_size, weight):
    # image shape bs*channel*h*w
    # 将图片分块
    # unfold其实就是stride=kernel_size的卷积
    # 按照下面参数，这里的patch的shape应该是1*48*4,然后transpose变为1*4*48
    # 1 是batch_size
    # 4 是patch块的个数
    # 48 是 8*8*3 / 4 ，即分割成4个patch后每个patch的大小
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
    # 矩阵相乘得到embedding矩阵
    patch_embedding = patch @ weight
    return patch_embedding


def image2emb_conv(image, kernel, stride):
    # b * c * h * w
    conv_output = F.conv2d(image, kernel, stride=stride)
    b, c, h, w = conv_output.shape
    # 把一张图片拉直成一个序列
    patch_embedding = conv_output.reshape(b, c, h * w).transpose(-1, -2)
    return patch_embedding


# vision transformer 论文实现
# 参数设定
parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=224, help="图片大小，默认224")
parser.add_argument("--patch_size", type=int, default=8, help="图片分块大小，默认8")
parser.add_argument("--batch_size", type=int, default=100, help="批次数量，一个批次100个样本训练")
parser.add_argument("--model_dim", type=int, default=224, help="模型维度，默认224")
parser.add_argument("--input_channel", type=int, default=3, help="输入通道数默认为3，即图片")
parser.add_argument("--epochs", type=int, default=10, help="训练轮数，默认10轮")
args = parser.parse_args()

# 其实就是h*w*c
patch_depth = args.patch_size * args.patch_size * args.input_channel
# 最大字符数，这里是（224/8） * （224 / 8） + 1 = 785
# +1 是因为第一个为cls token 即分类token
max_num_token = int((args.image_size / args.patch_size) ** 2 + 1)

# 定义数据规范化
data_transform = {
    # 训练
    "train": transforms.Compose([
        # 随机剪裁到224*224的大小
        transforms.RandomResizedCrop(args.image_size),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 张量化
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    # 测试
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
# 训练数据集，使用cifar10
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=data_transform["train"])
train_loader = DataLoader(train_dataset, batch_size=100,
                          shuffle=True, num_workers=0)
# 测试数据集
test_dataset = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=data_transform["test"])
test_loader = DataLoader(test_dataset, batch_size=100,
                         shuffle=True, num_workers=0)

# model_dim 是输出通道数(oc)，patch_depth 是卷积核的面积乘以输入通道数（ic）
weight = torch.randn(patch_depth, args.model_dim)
# 定义transformer模型
encoder_layer = nn.TransformerEncoderLayer(d_model=args.model_dim
                                           , nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=0.001)
# 训练后参数保存路径
save_path = "./params/vt.pth"


def train():
    # 训练
    for epoch in range(args.epochs):
        transformer_encoder.train()
        # 保存这一周期的损失值
        running_loss = 0.0
        # 保存训练开始时间
        t1 = time.perf_counter()
        for idx, data in enumerate(train_loader):
            # 解析数据以及标签部分
            images, labels = data
            # 清空梯度，避免叠加
            optimizer.zero_grad()
            # oc * ic * kh * kw
            kernel = weight.transpose(0, 1).reshape(-1, images.shape[1], args.patch_size, args.patch_size)
            # 二维卷积得到embedding
            pe2 = image2emb_conv(images, kernel, args.patch_size)
            # 第二步 在序列开头加一个可学习的embedding(cls token embedding)，用来做分类任务
            cls_token_embedding = torch.randn(images.shape[0], 1, args.model_dim, requires_grad=True)
            # 在位置上拼，所以dim=1
            token_embedding = torch.cat([cls_token_embedding, pe2], dim=1)
            # 第三步，位置编码(position embedding)
            position_embedding_table = torch.randn(max_num_token, args.model_dim,
                                                   requires_grad=True)
            # 序列长度
            seq_len = token_embedding.shape[1]  # 755
            position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
            token_embedding += position_embedding
            # 第四步，将embedding送入transformer的encoder中
            encoder_output = transformer_encoder(token_embedding)
            # 第五步，进行分类
            cls_token_output = encoder_output[:, 0, :]
            linear_layer = nn.Linear(args.model_dim, len(train_dataset.classes))
            logits = linear_layer(cls_token_output)
            # 计算损失
            loss = criterion(logits, labels)
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()
            # 叠加损失函数
            running_loss += loss
            # 保存当前进度
            rate = (idx + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = '.' * int((1 - rate) * 50)
            # 输出训练信息
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".
                  format(int(rate * 100),
                         a, b, loss), end='')
            # 输出本次训练耗时
        print('\n', time.perf_counter() - t1)


train()
