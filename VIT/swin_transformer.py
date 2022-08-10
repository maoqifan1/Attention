import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""
# 1、 如何基于图片生成patch embedding？
## 方法一
* 基于unfold api 来将图片进行分块，也就是仿卷积的思路，设置kernel_size=stride=patch_size,得到分块后的图片
* 得到格式为[bs, num_patch, patch_depth]的张量
* 将张量与形状为[patch_depth, model_dim_C]的权重矩阵进行乘法操作，即可得到形状为
  [bs, num_patch, model_dim_C]的patch embedding
## 方法二
* patch_depth 是等于input_channel*patch_size*patch_size
* model_dim_C相当于二维卷积的输出通道数目
* 将形状为[patch_depth, model_dim_C]的权重矩阵转换为[model_dim_C, input_channel, patch_size, patch_size]的卷积核
* 调用conv2d API得到卷积的输出张量，形状为[bs, output_channel, height, width]
* 转换为[bs, num_path, model_dim_C]格式，即为patch embedding
"""


def image2emb_naive(image, patch_size, weight):
    """
    :param image: (b, c, h, w)
    :param patch_size: 分块大小
    :param weight: 映射权重
    :return:
    """
    # [bs, num_patch,  patch_depth]
    patch = F.unfold(image, kernel_size=(patch_size, patch_size),
                     stride=(patch_size, patch_size)).transpose(-1, -2)
    # 矩阵乘法
    patch_embedding = patch @ weight  # [bs, num_patch, model_dim_C]
    return patch_embedding


def image2emb_conv(image, kernel, stride):
    """
    基于二维卷积实现patch embedding, embedding的维度就是卷积的输出通道数
    :param image: [b, c, h, w]
    :param kernel: 将形状为[patch_depth, model_dim_C]的权重矩阵转换为[model_dim_C, input_channel, patch_size, patch_size]的卷积核
    :param stride: 等于patch_size
    :return:
    """
    conv_output = F.conv2d(image, kernel, stride=stride)
    bs, oc, oh, ow = conv_output
    patch_embedding = conv_output.reshape((bs, oc, oh * ow)).transpose(-1, -2)  # [bs, num_patch, model_dim_C]
    return patch_embedding


"""
# 2、如何构建MultiHead Self-Attention并计算其复杂度
* 基于输入x进行三个映射，分别得到q, k, v
 * 此步的复杂度为3L*C^2 , 其中L为序列长度，C为特征大小
* 将q, k, v 拆成多头的形式，注意这里的多头各自计算不影响，所以可以与bs维度进行统一看待
* 计算qk^T，并考虑可能的掩码，即让无效的两两位置之间的能量为负无穷(-inf)，掩码在shift window MHSA中会需要
  在 window MHSA中暂时不需要
  * 此步复杂度为L^2*C
* 计算概率值与v的乘积
  * 此步复杂度为L^2*C
* 对输出进行再次映射
  * 此步复杂度为L*C^2
* 总体复杂度为4L*C^2 + 2L^2*C
"""


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_head=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        # 利用chunk api分别将输入投影到q，k，v
        self.proj_linear_layer = nn.Linear(model_dim, model_dim * 3)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, input_img, additive_mask=None):
        bs, seq_len, model_dim = input_img.shape
        num_head = self.num_head
        head_dim = model_dim // num_head  # 每个头的维度

        proj_output = self.proj_linear_layer(input_img)  # [bs, seq_len, model_dim * 3]
        q, k, v = proj_output.chunk(3, dim=-1)  # 3 * [bs, seq_len, model_dim]

        q = q.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2)  # [bs, num_head, seq_len, head_dim]
        q = q.reshape(bs * num_head, seq_len, head_dim)

        k = k.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2)  # [bs, num_head, seq_len, head_dim]
        k = k.reshape(bs * num_head, seq_len, head_dim)

        v = v.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2)  # [bs, num_head, seq_len, head_dim]
        v = v.reshape(bs * num_head, seq_len, head_dim)

        if additive_mask is None:
            # 得到一个概率，这个概率接下来会和v相乘得到output
            attn_prob = F.softmax(
                torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim),
                dim=-1
            )
        else:
            additive_mask = additive_mask.tile((num_head, 1, 1))
            # 算概率
            attn_prob = F.softmax(
                torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim) + additive_mask,
                dim=-1
            )
        # 概率值乘以v
        output = torch.bmm(attn_prob, v)  # [bs, num_head, seq_len, head_dim]
        output = output.reshape(bs, num_head, seq_len, head_dim).transpose(1, 2)  # [bs, seq_len, num_head, head_dim]
        # 将num_head和head_dim相乘，最后得到model_dim
        output = output.reshape(bs, seq_len, model_dim)  # [bs, seq_len, model_dim]
        # 最后通过线性层
        output = self.final_linear_layer(output)
        # 返回概率值，以及最后的结果
        return attn_prob, output


"""
# 3、如何构建Window MHSA并计算其复杂度
* 将patch组成的图片进一步划分成一个更大的window
  * 首先需要将三维的patch embedding转换成图片格式
  * 使用unfold将patch划分成window
* 在每个window内部计算MHSA
  * window数目其实可以和batch_size进行统一对待，因为window与window之间没有交互计算
  * 关于计算复杂度
    * 假设窗的边长为W，那么计算每个窗的总体复杂度为4W^2*C^2 + 2W^4*C
    * 假设patch的总数目为L，那么窗的数目为L/W^2
    * 因此，W-MHSA的总体复杂度为4L*C^2 + 2L*W^2*C
  * 此时，不需要mask
  * 将结果转化成带window的四维张量格式
* 复杂度对比
  * MHSA: 4L*C^2 + 2L^2*C
  * W-MHSA: 4L*C^2 + 2L*W^2*C
"""


def window_multi_head_self_attention(patch_embedding, mhsa, window_size=4, num_head=2):
    """
    :param patch_embedding: 三维张量 [bs, num_patch, patch_depth]
    :param mhsa: 多头自注意力
    :param window_size: 每个window的边长
    :param num_head: multi-head有多少个头
    :return:
    """
    num_patch_in_window = window_size * window_size
    bs, num_patch, patch_depth = patch_embedding.shape
    # 得到图片的高度和宽度
    image_height = image_width = int(math.sqrt(num_patch))
    patch_embedding = patch_embedding.transpose(-1, -2)  # [bs, patch_depth, num_patch]
    # 将embedding转换成图片格式
    patch = patch_embedding.reshape(bs, patch_depth, image_height, image_width)
    window = F.unfold(patch, kernel_size=(window_size, window_size),
                      stride=(window_size, window_size)).transpose(-1, -2)  # [bs, num_window, window_depth]
    bs, num_window, patch_depth_times_num_patch_in_window = window.shape
    # [bs*num_window, num_patch_in _window, patch_depth]
    window = window.reshape(bs * num_window, patch_depth, num_patch_in_window).transpose(-1, -2)

    attn_prob, output = mhsa(window)  # [bs*num_window, num_patch_in_window, patch_depth]
    # 把 bs*num_window 拆开成两个维度
    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    return output


"""
# 4、如何构建Shift Window  MHSA及其 MasK
* 将上一部的W-MHSA的结果转换成图片格式。
* 假设已经做了新的window划分，这一步叫shift-window
* 为了保持window数目不变从而高效计算，需要将图片的patch往左和往上各自滑动半个窗口大小的步长，保持patch所属window类别不变
* 将图片patch还原为window的数据格式
* 由于cycle shift后，每个window虽然形状规整，但部分window中存在原本不属于同一个窗口的patch，所以需要生成mask
* 如何生成mask?
  * 首先构建一个shift-window的patch所属的window类别矩阵
  * 对该矩阵进行同样的往左和往上各自滑动半个窗口大小的步长的操作
  * 通过unfold操作得到[bs, num_window, num_patch_in_window]形状的类别矩阵
  * 对该矩阵扩维成[bs, num_window, num_patch_in_window, 1]
  * 将该矩阵与其转置矩阵进行做差，得到同类关系矩阵(为0的位置上的patch属于同类，否则属于不同类)
  * 对同类矩阵中非零的未知用负无穷数进行填充，对于零的位置用0去填充, 这样就构建好了MHSA所需要的mask
  * 此mask的形状为[bs, num_window, num_patch_in_window, num_patch_in_window]
* 将window转换成三维的格式，[bs*num_window, num_patch_in_window,patch_depth]
* 将三维格式的特征连同mask一起送入MHSA中计算得到注意力输出
* 将注意力输出转换成图片patch格式，[bs, num_window,  num_patch_in_window, patch_depth]
* 为了恢复图片原来的位置，还需要将图片的patch往右和往下各自滑动半个窗口大小的步长，至此，SW-MHSA计算完毕

"""


def window2image(mhsa_output):
    """
    辅助函数，将transformer block的结果转化成图片的格式
    :param mhsa_output:  window_multi_head_self_attention()函数的输出
    :return:
    """
    bs, num_window, num_patch_in_window, patch_depth = mhsa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_height = int(math.sqrt(num_window)) * window_size
    image_width = image_height
    # sqrt(num_window) *  sqrt(num_window) = num_window
    # window_size * window_size = num_patch_in_window
    mhsa_output = mhsa_output.reshape(bs, int(math.sqrt(num_window)),
                                      int(math.sqrt(num_window)),
                                      window_size,
                                      window_size,
                                      patch_depth)
    mhsa_output = mhsa_output.transpose(2, 3)
    image = mhsa_output.reshape(bs, image_height * image_width, patch_depth)
    # 保持与卷积格式一致
    image = image.transpose(-1, -2).reshape(bs, patch_depth, image_height, image_width)
    return image


def shift_window(w_mhsa_output, window_size, shift_size, generate_mask=False):
    """
    :param w_mhsa_output: window_multi_head_self_attention()函数的输出
    :param window_size:
    :param shift_size:
    :param generate_mask:
    :return:
    """
    bs, num_window, num_patch_in_window, patch_depth = w_mhsa_output.shape
    # 转换成图片格式
    w_mhsa_output = window2image(w_mhsa_output)  # [bs, depth, height, width]
    bs, patch_depth, image_height, image_width = w_mhsa_output.shape
    # 向左向上roll
    rolled_w_mhsa_output = torch.roll(w_mhsa_output, shifts=(shift_size, shift_size),
                                      dims=(2, 3))  # image_height, image_width

    shifted_w_mhsa_input = rolled_w_mhsa_output.reshape(bs, patch_depth,
                                                        int(math.sqrt(num_window)),
                                                        window_size,
                                                        int(math.sqrt(num_window)),
                                                        window_size
                                                        )

    shifted_w_mhsa_input = shifted_w_mhsa_input.transpose(3, 4)
    shifted_w_mhsa_input = shifted_w_mhsa_input.reshape(bs, patch_depth
                                                        , num_window * num_patch_in_window)
    shifted_w_mhsa_input = shifted_w_mhsa_input.transpose(-1, -2)  # [bs, num_window*num_patch_in_window, patch_depth]

    shifted_w_mhsa_input = shifted_w_mhsa_input.reshape(bs, num_window, num_patch_in_window, patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmhsa(batch_size=bs,
                                                     image_height=image_height,
                                                     image_width=image_width,
                                                     window_size=window_size)
    else:
        additive_mask = None
    return shifted_w_mhsa_input, additive_mask


def build_mask_for_shifted_wmhsa(batch_size, image_height, image_width, window_size):
    """
    构建shift window multi-head attention mask
    :param batch_size:
    :param image_height:
    :param image_width:
    :param window_size:
    :return:
    """
    index_matrix = torch.zeros(image_height, image_width)

    for i in range(image_height):
        for j in range(image_width):
            row_times = (i + window_size // 2) // window_size
            col_times = (j + window_size // 2) // window_size
            index_matrix[i, j] = row_times * (image_height // window_size) + col_times + 1
    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size // 2, -window_size // 2), dims=(0, 1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0)  # [bs, ch , h, w]

    c = F.unfold(rolled_index_matrix, kernel_size=(window_size, window_size),
                 stride=(window_size, window_size)).transpose(-1, -2)
    c = c.tile(batch_size, 1, 1)  # [bs, num_window, num_patch_in_window]

    bs, num_window, num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1)  # [bs, num_window, num_patch_in_window, 1]
    c2 = (c1 - c1.transpose(-1, -2)) == 0  # [bs, num_window, num_patch_in_window, num_patch_in_window]
    valid_matrix = c2.to(torch.float32)
    additive_mask = (1 - valid_matrix) * (-1e9)  # [bs, num_window, num_patch_in_window, num_patch_in_window]

    additive_mask = additive_mask.reshape(bs * num_window, num_patch_in_window, num_patch_in_window)

    return additive_mask


def shift_window_multi_head_self_attention(w_mhsa_output, mhsa, window_size=4, num_head=2):
    """
    :param w_mhsa_output:
    :param mhsa:
    :param window_size:
    :param num_head:
    :return:
    """
    bs, num_window, num_patch_in_window, patch_depth = w_mhsa_output.shape
    # shift
    shifted_w_mhsa_input, additive_mask = shift_window(w_mhsa_output,
                                                       window_size,
                                                       shift_size=(-window_size) // 2,
                                                       generate_mask=True)
    # shifted_w_mhsa_input [bs, num_window, num_patch_in_window, patch_depth]
    # additive_mask [bs*num_window, num_patch_in_window, num_patch_in_window]

    shifted_w_mhsa_input = shifted_w_mhsa_input.reshape(bs * num_window, num_patch_in_window, patch_depth)

    attn_prob, output = mhsa(shifted_w_mhsa_input, additive_mask=additive_mask)

    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    # 反向 shift
    output, _ = shift_window(output, window_size, shift_size=window_size // 2, generate_mask=False)
    # output [bs, num_window, num_patch_in_window, patch_depth]
    return output


"""
# 5、如何构建 Patch Merging
* 将window格式的特征转换为图片patch格式
* 利用unfold操作，按照merge_size*merge_size的大小得到新的patch，形状为[bs, num_patch_new, merge_size*merge_size*patch_depth_old]
* 使用一个全连接层对depth进行降维成0.5倍，也就是merge_size*merge_size*patch_depth_old 映射到 0.5*merge_size*merge_size*patch_depth_old
* 输出的是patch embedding的形状格式,[bs, num_patch, patch_depth]
* 举例：以merge_size=2为例，经过PatchMerging后，patch数目减少为原来的1/4，但是depth增大为原来的2倍，而不是4倍
"""


class PatchMerging(nn.Module):
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(in_features=model_dim * merge_size * merge_size,
                                    out_features=int(model_dim * merge_size * merge_size * output_depth_scale))

    def forward(self, input_val):
        bs, num_window, num_patch_in_window, patch_depth = input_val.shape
        window_size = int(math.sqrt(num_patch_in_window))

        input_val = window2image(input_val)  # [bs, patch_depth, image_h,image_w]

        merged_window = F.unfold(input_val, kernel_size=(self.merge_size, self.merge_size),
                                 stride=(self.merge_size, self.merge_size)).transpose(-1, -2)

        merged_window = self.proj_layer(merged_window)

        return merged_window


"""
# 6、如何构建SwinTransformerBlock
* 每个block包含LayerNorm、W-MHSA、SW-MHSA、残差链接等模块
* 输入是patch embedding格式
* 每个MLP包含两层，分别是4*model_dim和model_dim大小
* 输出的是window的数据格式,[bs, num_window, num_patch_in_window, patch_depth]
* 需要注意残差链接对数据形状的要求
"""


class SwinTransformerBlock(nn.Module):
    def __init__(self, model_dim, window_size, num_head):
        super(SwinTransformerBlock, self).__init__()

        self.window_size = window_size
        self.num_head = num_head
        # 层归一化
        self.layer_nom1 = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_nom2 = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_nom3 = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_nom4 = nn.LayerNorm(normalized_shape=model_dim)

        # mlp(线性层)
        # for w-mhsa
        self.wmhsa_mlp1 = nn.Linear(model_dim, model_dim * 4)
        self.wmhsa_mlp2 = nn.Linear(4 * model_dim, model_dim)
        # for sw-mhsa
        self.s_wmhsa_mlp1 = nn.Linear(model_dim, model_dim * 4)
        self.s_wmhsa_mlp2 = nn.Linear(4 * model_dim, model_dim)

        # multi-head self attention
        # for window and shift-window
        self.mhsa1 = MultiHeadSelfAttention(model_dim, num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim, num_head)

    def forward(self, input_val):
        bs, num_patch, patch_depth = input_val.shape

        input1 = self.layer_nom1(input_val)
        w_mhsa_output = window_multi_head_self_attention(input1, self.mhsa1,
                                                         window_size=self.window_size,
                                                         num_head=self.num_head)
        bs, num_window, num_patch_in_window, patch_depth = w_mhsa_output.shape
        w_mhsa_output = input1 + w_mhsa_output.reshape(bs, num_patch, patch_depth)
        output1 = self.wmhsa_mlp2(self.wmhsa_mlp1(self.layer_nom2(w_mhsa_output)))
        output1 += w_mhsa_output

        input2 = self.layer_nom3(output1)
        input2 = input2.reshape(bs, num_window, num_patch_in_window, patch_depth)
        sw_mhsa_output = shift_window_multi_head_self_attention(input2, self.mhsa2,
                                                                window_size=self.window_size
                                                                , num_head=self.num_head)
        sw_mhsa_output = output1 + sw_mhsa_output.reshape(bs, num_patch, patch_depth)
        output2 = self.s_wmhsa_mlp2(self.s_wmhsa_mlp1(self.layer_nom4(sw_mhsa_output)))
        output2 += sw_mhsa_output

        output2 = output2.reshape(bs, num_window, num_patch_in_window, patch_depth)

        return output2


class SwinTransformerModel(nn.Module):
    def __init__(self, input_image_channel1=3, patch_size=4, model_dim_c=8,
                 num_classes=10, window_size=4, num_head=2, merge_size=2):
        super(SwinTransformerModel, self).__init__()
        patch_depth = patch_size * patch_size * input_image_channel1
        self.patch_size = patch_size
        self.model_dim_C = model_dim_c
        self.num_classes = num_classes

        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_c))
        self.block1 = SwinTransformerBlock(model_dim=model_dim_c, window_size=window_size, num_head=num_head)
        self.block2 = SwinTransformerBlock(model_dim_c * 2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_c * 4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_c * 8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_c, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_c * 2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_c * 4, merge_size)

        self.final_layer = nn.Linear(model_dim_c * 8, num_classes)

    def forward(self, image):
        # 图片分块
        # [bs, num_patch, model_dim_C]
        patch_embedding_naive = image2emb_naive(image, self.patch_size, self.patch_embedding_weight)

        # block1
        patch_embedding = patch_embedding_naive
        print("patch_embedding{}".format(patch_embedding.shape))

        sw_mhsa_output = self.block1(patch_embedding)  # [bs, num_window,num_patch_in_window, patch_depth]
        print("block1_output{}".format(sw_mhsa_output.shape))

        merged_patch1 = self.patch_merging1(sw_mhsa_output)
        sw_mhsa_output1 = self.block2(merged_patch1)
        print("block2_output{}".format(sw_mhsa_output1.shape))

        merged_patch2 = self.patch_merging2(sw_mhsa_output1)
        sw_mhsa_output2 = self.block3(merged_patch2)
        print("block3_output{}".format(sw_mhsa_output2.shape))

        merged_patch3 = self.patch_merging3(sw_mhsa_output2)
        sw_mhsa_output3 = self.block4(merged_patch3)
        print("block4_output{}".format(sw_mhsa_output3.shape))

        bs, num_window, num_patch_in_window, patch_depth = sw_mhsa_output3.shape
        sw_mhsa_output3 = sw_mhsa_output3.reshape(bs, -1, patch_depth)
        # 池化
        pool_output = torch.mean(sw_mhsa_output3, dim=1)
        logits = self.final_layer(pool_output)
        print("logits{}".format(logits.shape))
        return logits


if __name__ == '__main__':
    batch_size, ic, image_h, image_w = 4, 3, 256, 256

    _image = torch.randn((batch_size, ic, image_h, image_w))
    model = SwinTransformerModel()
    model(_image)
