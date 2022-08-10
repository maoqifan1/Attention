import torch
import torch.nn as nn
import torch.nn.functional as F


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


# 测试image2emb的代码
batch_size, input_channel, image_h, image_w = 1, 3, 8, 8
# 即一个4*4为一个patch
patch_size = 4
# 其实就是h*w*c
patch_depth = patch_size * patch_size * input_channel
model_dim = 8
max_num_token = 16
image = torch.randn((batch_size, input_channel, image_h, image_w))
# model_dim 是输出通道数(oc)，patch_depth 是卷积核的面积乘以输入通道数（ic）
weight = torch.randn(patch_depth, model_dim)
# 分块方法得到embedding
pe1 = image2emb_naive(image, patch_size, weight)
print(pe1.shape)
# oc * ic * kh * kw
kernel = weight.transpose(0, 1).reshape(-1, input_channel, patch_size, patch_size)
# 二维卷积得到embedding
pe2 = image2emb_conv(image, kernel, patch_size)
print(pe2.shape)

# 第二步，在序列开头加一个可学习的embedding(cls token embedding)，用来做分类任务
cls_token_embedding = torch.randn(batch_size, 1, model_dim, requires_grad=True)
token_embedding = torch.cat([cls_token_embedding, pe2], dim=1)

# 第三步，位置编码(position embedding)
position_embedding_table = torch.randn(max_num_token, model_dim, requires_grad=True)
seq_len = token_embedding.shape[1]  # 5

position_embedding = torch.tile(position_embedding_table[:seq_len], [token_embedding.shape[0], 1, 1])
token_embedding += position_embedding

# 第四步，将embedding送入transformer的encoder中
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim
                                           , nhead=8)
