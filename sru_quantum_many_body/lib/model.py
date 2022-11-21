# -*- coding: utf-8 -*-
"""
Created on 2019/4/8

@author: Robin

Full Transformer Model
"""
import torch
import torch.nn as nn

from transformer.modules import FeatureEncoding, SystemEncoding, RestoreEmbedding, MultiHeadAttention, FeedForward


class BetaTransformer(nn.Module):
    def __init__(self, hp):
        """
        :param hp: Hyper Parameters
        """
        super(BetaTransformer, self).__init__()
        self.hp = hp

        # Encoder
        self.enc_fe = FeatureEncoding(max_features=self.hp.max_features, num_units=self.hp.d_model)
        if self.hp.system_encoding:
            self.enc_se = SystemEncoding(max_particle=self.hp.max_particle, num_units=self.hp.d_model)
        self.enc_dp = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_multihead_att_{}'.format(i), MultiHeadAttention(num_units=self.hp.d_model,
                                                                                  num_heads=self.hp.num_heads,
                                                                                  dropout_rate=self.hp.dropout_rate,
                                                                                  causality=False))
            self.__setattr__('enc_feedforward_{}'.format(i), FeedForward(num_units=self.hp.d_model,
                                                                         num_hidden_units=self.hp.d_model * 4,
                                                                         dropout_rate=hp.dropout_rate))

        # Decoder
        self.dec_fe = FeatureEncoding(max_features=self.hp.max_features, num_units=self.hp.d_model)
        if self.hp.system_encoding:
            self.dec_se = SystemEncoding(max_particle=self.hp.max_particle, num_units=self.hp.d_model)
        self.dec_dp = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_multihead_att_{}'.format(i), MultiHeadAttention(num_units=self.hp.d_model,
                                                                                  num_heads=self.hp.num_heads,
                                                                                  dropout_rate=self.hp.dropout_rate,
                                                                                  causality=True))
            self.__setattr__('dec_cross_att_{}'.format(i), MultiHeadAttention(num_units=self.hp.d_model,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('dec_feedforward_{}'.format(i), FeedForward(num_units=self.hp.d_model,
                                                                         num_hidden_units=self.hp.d_model * 4,
                                                                         dropout_rate=hp.dropout_rate))

        self.dec_re = RestoreEmbedding(max_features=self.hp.max_features, num_units=self.hp.d_model)

    def forward(self, x, y, num_particle):
        """
        :param x: encoder input. (B, T_e, ReP + ImP)
        :param y: decoder input. (B, T_d, ReP + ImP)
        :param num_particle: int: number of particles in this batch
        :return: loss, pred
        """
        assert x.size(-1) == 2 ** (num_particle + 1)

        # Encoder
        x = self.enc_fe(x)  # (B, T_e, D)
        if self.hp.system_encoding:
            x += self.enc_se(x, num_particle)  # (B, T_e, D)  # to(hp.device) in modules
        x = self.enc_dp(x)

        for i in range(self.hp.num_blocks):
            x = self.__getattr__('enc_multihead_att_{}'.format(i))(x, x, x)
            x = self.__getattr__('enc_feedforward_{}'.format(i))(x)  # (B, T_e, D)

        # Decoder
        # 第一个time step置0(offset by one position)
        y_ = torch.cat((torch.ones_like(y[:, :1, :]).to(self.hp.device), y[:, :-1, :]), -2)  # (B, T_d, ReP + ImP)
        # y_ = torch.cat((torch.zeros_like(y[:, :1, :]).to(self.hp.device), y[:, :-1, :]), -2)
        y_ = self.dec_fe(y_)  # (B, T_d, D)
        if self.hp.system_encoding:
            y_ += self.dec_se(y_, num_particle)
        y_ = self.dec_dp(y_)

        for i in range(self.hp.num_blocks):
            y_ = self.__getattr__('dec_multihead_att_{}'.format(i))(y_, y_, y_)
            y_ = self.__getattr__('dec_cross_att_{}'.format(i))(y_, x, x)
            y_ = self.__getattr__('dec_feedforward_{}'.format(i))(y_)

        y_ = self.dec_re(y_, num_particle)  # (B, T_d, ReP + ImP)

        # Loss
        loss = nn.MSELoss()(y, y_)

        return loss, y_
