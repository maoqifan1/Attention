import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

# 关于word embedding，以序列建模为例
# 考虑 source sentence和target sentence
# 构建序列，序列的字符以其在词表中的索引的形式表示
# 批度大小
batch_size = 2
# 单词表大小
max_num_src_words = 8
max_num_tgt_words = 8

# 序列最大长度
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5

# 模型的特征大小
model_dim = 8

# 每一个值代表句子中的单词数，有两个句子
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# 利用F.pad对序列进行padding，从长度不足的部分开始，max_src_seq_len - L就是这个意思
# torch.unsqueeze()增加一个维度，然后把两个句子拼接起来
# 则第一行就是第一个句子
# 第二行就是第二个句子
# padding后的句子长度就全部变成序列最大长度5的句子了
src_seq = torch.cat([torch.unsqueeze
                     (F.pad(torch.randint(1, max_num_src_words, (L,)),
                            (0, max(src_len) - L),
                            ), 0)  # 在第0维添加一个维度
                     for L in src_len])

# 这里进行和src_seq一样的操作
tgt_seq = torch.cat([torch.unsqueeze(
    F.pad(torch.randint(1, max_num_tgt_words, (L,)),
          (0, max(src_len) - L))
    , 0)
    for L in tgt_len])

# 构造word_embedding
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)
# 将句子进行嵌入
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)

# 2. 构建 position embedding
pos_mat = torch.arange(max_position_len).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1)) / model_dim)
pe_embedding_table = torch.zeros(max_position_len, model_dim)
pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# 进行嵌入
pe_embedding = nn.Embedding(max_position_len, model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)

# 3. 构建位置索引
src_pos = torch.cat([
    torch.unsqueeze(torch.arange(max(src_len)), 0) for _ in src_len
]).to(torch.int32)
tgt_pos = torch.cat([
    torch.unsqueeze(torch.arange(max(tgt_len)), 0) for _ in tgt_len
]).to(torch.int32)

# 注意传入pe_embedding的应该是位置索引而不是单词索引
src_pe_embedding = pe_embedding(src_pos)
tgt_pe_embedding = pe_embedding(tgt_pos)

# 4. 构造encoder的self-attention mask
# mask的shape:[batch_size, max_src_len, max_src_len], 值为1或-inf
# 得到输入序列的有效位置
# 对其进行扩充以达到最大序列长度
valid_encoder_pos = torch.unsqueeze(
    torch.cat([
        torch.unsqueeze(
            F.pad(torch.ones(L), (0, max(src_len) - L))
            , 0)
        for L in src_len])
    , 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix

# 将0转为False，1转为True
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)

encoder_score = torch.randn(batch_size, max(src_len), max(src_len))
encoder_masked_score = encoder_score.masked_fill(mask_encoder_self_attention, -np.inf)

# 5.构造intra-attention的mask
# Q * k^T shape : [batch_size, tgt_seq_len, src_seq_len]
valid_decoder_pos = torch.unsqueeze(
    torch.cat([
        torch.unsqueeze(
            F.pad(torch.ones(L), (0, max(tgt_len) - L))
            , 0)
        for L in tgt_len])
    , 2)
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)

# 6. 构造decoder self-attention的mask
valid_decoder_tri_matrix = torch.cat([
    torch.unsqueeze(
        F.pad(torch.tril(torch.ones((L, L))),
              (0, max(tgt_len) - L, 0, max(tgt_len) - L))
        , 0) for L in
    tgt_len])
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)
decoder_score = torch.randn(batch_size, max(tgt_len), max(tgt_len))
decoder_masked_score = decoder_score.masked_fill(invalid_decoder_tri_matrix, -1e9)
prob = F.softmax(decoder_masked_score, -1)
print(prob)


# 7. 构建self-attention
def scaled_dot_product(Q, K, V, attn_mask):
    # shape of Q、K、V:(batch_size*num_heads, seq_len, model_dim / num_heads)
    score = torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask, -1e9)
    F.softmax(masked_score)
    context = torch.bmm(prob, V)
    return context
