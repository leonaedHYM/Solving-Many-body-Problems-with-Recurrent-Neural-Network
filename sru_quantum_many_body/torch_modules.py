#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : torch_modules.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 22.07.2019
# Last Modified Date: 18.08.2019
# Last Modified By  : zewangzhang <zewangzhang@tencent.com>
# -*- coding: utf-8 -*-
"""
Created on 2019/4/8

@author: Robin

Necessary modules for Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparams import hp


class FeatureEncoding(nn.Module):
    """
    Feature Encoding (特征编码) --- 将不同粒子数系统的特征向量映射到统一维度
    即(B, T, ReP + ImP) -> (B, T, D/2 + D/2) (实部与虚部分开处理)

    *注: embedding所需要的值溢出查询表的话(>max_features),返回的是无意义的0值
    """

    def __init__(self, max_features, num_units):
        """
        :param max_features: 实部(或虚部)查询字典的键的数目
        :param num_units: 统一后的维度: D
        """
        super(FeatureEncoding, self).__init__()
        self.num_units = num_units  # d_model = 512
        # [500, 256]
        self.lookup_table_real = nn.Parameter(torch.Tensor(max_features, int(num_units / 2)))  # (max_features, D/2)
        self.lookup_table_imag = nn.Parameter(torch.Tensor(max_features, int(num_units / 2)))  # (max_features, D/2)
        nn.init.xavier_normal_(self.lookup_table_real)
        nn.init.xavier_normal_(self.lookup_table_imag)

    def forward(self, inputs):
        # inputs:[100,500,64]
        batch_size, steps = inputs.size(0), inputs.size(1)
        cur_fea = int(inputs.size(2) / 2)

        # [100, 500, 32]
        base = torch.arange(0, cur_fea).repeat(batch_size, steps, 1).long().to(hp.device)

        # [100,500,32,256]
        real_base = F.embedding(base, self.lookup_table_real)
        imag_base = F.embedding(base, self.lookup_table_imag)

        # [100,500,32,1]
        real_inputs = torch.unsqueeze(inputs[:, :, :int(cur_fea)], -1)
        imag_inputs = torch.unsqueeze(inputs[:, :, int(cur_fea):], -1)

        # [100,500,32,256]
        real_weighted = real_base * real_inputs
        imag_weighted = imag_base * imag_inputs

        # [100,500,256]
        real_reduced = torch.sum(real_weighted, -2)  # axis=-2位置求和收缩
        imag_reduced = torch.sum(imag_weighted, -2)

        # [100,500,512]
        outputs = torch.cat((real_reduced, imag_reduced), -1)

        return outputs


class SystemEncoding(nn.Module):
    """
    System Encoding (系统编码/粒子数编码) --- An encoding that distinguish different systems

    output: An encoding of (B, T, D), which should be added to the result from Feature Encoding
    """

    def __init__(self, max_particle, num_units):
        """
        :param max_particle: 查询字典的键的数目(最大允许粒子数)
        :param num_units: 输出张量的最后一个维度大小: hp.d_model
        """
        super(SystemEncoding, self).__init__()
        self.num_units = num_units
        self.lookup_table = nn.Parameter(torch.Tensor(max_particle + 1, num_units))  # +1: 不包括0; 前两行是没用的(没有0, 1)
        nn.init.xavier_normal_(self.lookup_table)

    def forward(self, inputs, num_particle):  # (B, T, D)
        batch_size, steps = inputs.size(0), inputs.size(1)

        base = torch.Tensor([num_particle]).long().to(hp.device)  # (1,)

        # [1, num_units]
        outputs = F.embedding(base, self.lookup_table)
        # [batch_size, steps, num_units]
        outputs = outputs.repeat(batch_size, steps, 1)  # (B, T, D)

        return outputs


class RestoreEmbedding(nn.Module):
    """
    恢复Embedding --- 将数据的最后一个维度恢复到这个batch中数据原本的维度 e.g. 512 -> 64(6粒子)
    最后将实部和虚部拼接起来,输出 (B, T, ReP + ImP)
    """

    def __init__(self, max_features=512, num_units=512):
        """
        :param max_features: 实部(或虚部)查询字典的键的数目
        :param num_units: 输入张量的最后一个维度大小: hp.d_model
        """
        super(RestoreEmbedding, self).__init__()
        self.num_units = num_units
        self.lookup_table_real = nn.Parameter(torch.Tensor(max_features, num_units))  # (max_features, D)
        self.lookup_table_imag = nn.Parameter(torch.Tensor(max_features, num_units))  # (max_features, D)
        nn.init.xavier_normal_(self.lookup_table_real)
        nn.init.xavier_normal_(self.lookup_table_imag)

    def forward(self, inputs, num_particle):  # (B, T, D)
        # inputs:[BS,TL,num_units]
        base = torch.arange(2 ** num_particle).long().to(hp.device)  # (实部或虚部原本的特征维度,)

        # [2**num_particles, num_units]
        real_base = F.embedding(base, self.lookup_table_real)  # (实部或虚部原本的特征维度, D)
        imag_base = F.embedding(base, self.lookup_table_imag)

        # [BS,TL,2**num_particles]
        real_part = torch.matmul(inputs, real_base.permute((1, 0)))  # (B, T, 实部或虚部原本的特征维度)
        imag_part = torch.matmul(inputs, imag_base.permute((1, 0)))

        # 实部虚部拼接
        outputs = torch.cat((real_part, imag_part), -1)  # (B, T, ReP + ImP)
        return outputs


class LayerNorm(nn.Module):
    """
    Layer Normalization
    """

    def __init__(self, num_units, epsilon=1e-8):
        """
        :param num_units: D
        :param epsilon: 1e-8
        """
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(num_units))  # (D,)
        self.beta = nn.Parameter(torch.zeros(num_units))

    def forward(self, inputs):  # (B, T, D)
        mean = inputs.mean(-1, keepdim=True)  # (B, T, 1)
        std = inputs.std(-1, keepdim=True)  # (B, T, 1)
        return self.gamma * (inputs - mean) / (std + self.epsilon) + self.beta


class MultiHeadAttention(nn.Module):
    """
    多头注意力层
    (B, T, D) -> (B, T, D)
    """

    def __init__(self, num_units, num_heads=8, dropout_rate=0.1, causality=False, zeros_pad=False, scale=True):
        """
        :param num_units: D
        :param num_heads: number of self-attention heads
        :param dropout_rate:
        :param causality:
        """
        super(MultiHeadAttention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.zeros_pad = zeros_pad
        self.scale = scale

        self.q_proj = nn.Linear(self.num_units, self.num_units)
        self.k_proj = nn.Linear(self.num_units, self.num_units)
        self.v_proj = nn.Linear(self.num_units, self.num_units)

        self.output_dropout = nn.Dropout(p=self.dropout_rate)
        self.norm = LayerNorm(self.num_units)

    def forward(self, q, k, v):
        # encoder 的 q, k, v均为输入inputs
        # decoder 的 q来自自己的输入, k, v为encoder的输出
        # q (B, T_q, D_q)
        # k, v (B, T_k, D_k)

        # Linear Projections
        q_ = self.q_proj(q)  # (B, T_q, D_q) multihead(h heads) : D_q / h * h
        k_ = self.k_proj(k)
        v_ = self.v_proj(v)

        # Split and Concatenate
        q_ = torch.cat(torch.chunk(q_, self.num_heads, dim=2), dim=0)  # (h * B, T_q, D / h)
        k_ = torch.cat(torch.chunk(k_, self.num_heads, dim=2), dim=0)  # (h * B, T_k, D / h)
        v_ = torch.cat(torch.chunk(v_, self.num_heads, dim=2), dim=0)

        # Multiplication
        outputs = torch.matmul(q_, k_.permute((0, 2, 1)))  # (h * B, T_q, T_k)

        # Scale
        if self.scale:
            outputs = outputs / (k_.size(-1) ** 0.5)

        # Zeros Pad (Key Masking)
        if self.zeros_pad:
            k_mask = torch.sign(torch.sum(torch.abs(k), dim=-1))  # (B, T_k)
            k_mask = k_mask.repeat(self.num_heads, 1)  # (h * B, T_k)
            k_mask = torch.unsqueeze(k_mask, 1).repeat(1, q_.size(1), 1)  # (h * B, T_q, T_k)

            padding = torch.ones_like(outputs).to(hp.device) * (-2 ** 32 + 1)
            condition = k_mask.eq(0).float()  # padding through '1' place
            outputs = padding * condition + outputs * (1 - condition)

        # Causality (Future Blinding in Decoder)
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size()).to(hp.device)  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # 保留下三角(包括对角线部分),其他置零 (T_q, T_k)
            masks = torch.unsqueeze(tril, 0).repeat(outputs.size(0), 1, 1)  # (h * B, T_q, T_k)

            padding = torch.ones_like(masks).to(hp.device) * (-2 * 32 + 1)  # (h * B, T_q, T_k)
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1 - condition)  # 未来信息变为-inf,其他位置不变

        outputs = F.softmax(outputs, -1)  # (h * B, T_q, T_k) outputs现在是表示概率信息的方阵

        # Zeros Pad (Query Mask)
        if self.zeros_pad:
            q_mask = torch.sign(torch.sum(torch.abs(q), -1))  # (B, T_q)
            q_mask = q_mask.repeat(self.num_heads, 1)  # (h * B, T_q)
            q_mask = torch.unsqueeze(q_mask, 2).repeat(1, 1, k.size(1))  # (h * B, T_q, T_k)
            outputs = outputs * q_mask

        # Dropout
        outputs = self.output_dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, v_)  # (h * B, T_q, D / h)

        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (B, T_q, D)

        # Residual connection and Layer Normalization
        outputs = self.norm(outputs + q)  # (B, T_q, D)
        return outputs


class FeedForward(nn.Module):
    """
    前向全连接层
    """

    def __init__(self, num_units, num_hidden_units, dropout_rate):
        """
        :param num_units: D
        :param num_hidden_units: D_ff (隐藏层维度)
        """
        super(FeedForward, self).__init__()
        self.num_units = num_units
        self.num_hidden_units = num_hidden_units
        self.dropout_rate = dropout_rate

        self.w1 = nn.Linear(self.num_units, self.num_hidden_units)
        self.w2 = nn.Linear(self.num_hidden_units, self.num_units)
        self.norm = LayerNorm(self.num_units)
        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        outputs = self.w2(F.relu(self.w1(inputs)))
        outputs = self.output_dropout(outputs)
        outputs = self.norm(outputs + inputs)
        return outputs
