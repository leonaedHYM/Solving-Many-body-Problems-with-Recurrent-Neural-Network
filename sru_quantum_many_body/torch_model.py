#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : torch_model.py
# Author            : zewangzhang <zzw922cn@gmail.com>
# Date              : 04.10.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-
# File              : torch_model.py
# Date              : 22.07.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch_modules import FeatureEncoding, SystemEncoding, RestoreEmbedding, FeedForward
from sru import SRU, SRUCell



class RNNModel(nn.Module):
    def __init__(self, hp, inference=False):
        """
        :param hp: Hyper Parameters
        """
        super(RNNModel, self).__init__()
        self.hp = hp
        dropout_rate = hp.dropout_rate
        if inference:
            dropout_rate = 0

        self.prenet_feature = FeatureEncoding(max_features=2**hp.max_particle, num_units=hp.feature_embedding_size)
        if hp.add_system_encoding:
            self.prenet_system = SystemEncoding(max_particle=hp.max_particle, num_units=hp.system_embedding_size)

        self.prenet_dense_layer = FeedForward(hp.system_embedding_size, hp.prenet_units, dropout_rate)
        self.prenet_dropout = nn.Dropout(dropout_rate)

        #self.rnn = nn.RNN(hp.system_embedding_size, hp.rnn_units, hp.rnn_layers, batch_first=True)
        print(hp.use_relu)
        self.rnn = SRU(hp.system_embedding_size, hp.rnn_units, hp.rnn_layers, dropout=dropout_rate, rnn_dropout=dropout_rate, use_relu=hp.use_relu, rescale=True, layer_norm=hp.layer_norm)

        if hp.add_system_encoding:
            self.postnet_system = SystemEncoding(max_particle=hp.max_particle, num_units=hp.system_embedding_size)

        self.h0_dense_layer = FeedForward(self.hp.system_embedding_size, self.hp.rnn_units, dropout_rate)
        self.postnet_dense_layer = FeedForward(self.hp.rnn_units, self.hp.postnet_units, dropout_rate)
        self.postnet_dropout = nn.Dropout(dropout_rate)
        self.postnet_restore = RestoreEmbedding(max_features=2**hp.max_particle, num_units=hp.postnet_units)

    def forward(self, x, y, num_particle):
        """
        :param x: encoder input. (B, T_e, ReP + ImP)
        :param y: decoder input. (B, T_d, ReP + ImP)
        :param num_particle: int: number of particles in this batch
        :return: loss, pred
        """
        assert x.size(-1) == 2 ** (num_particle + 1)

        # [BS, TL, feature_embedding_size]
        x = self.prenet_feature(x)
        if self.hp.add_system_encoding:
            x += self.prenet_system(x, num_particle)  # (B, T_e, D)  # to(hp.device) in modules
        x = self.prenet_dense_layer(x)
        x = self.prenet_dropout(x)

        # (num_layers * num_directions, batch, hidden_size)
        rnn_h0 = self.h0_dense_layer(self.prenet_system(x, num_particle)[0, 0, :]).repeat(self.hp.rnn_layers, x.size(0), 1)

        x = x.permute(1, 0, 2)
        rnn_output, hn = self.rnn(x, rnn_h0)

        rnn_output = rnn_output.permute(1, 0, 2)
        postnet_output = self.postnet_dense_layer(rnn_output)
        postnet_output = self.postnet_dropout(postnet_output)

        y_ = self.postnet_restore(postnet_output, num_particle)  # (B, T_d, ReP + ImP)

        # Loss
        loss = nn.MSELoss()(y, y_)

        return loss, y_

    def generate(self, x, num_particle, rnn_h0=None):

        x = self.prenet_feature(x)
        if self.hp.add_system_encoding:
            x += self.prenet_system(x, num_particle)  # (B, T_e, D)  # to(hp.device) in modules
        x = self.prenet_dense_layer(x)
        x = self.prenet_dropout(x)

        # (num_layers * num_directions, batch, hidden_size)
        if rnn_h0 is None:
            rnn_h0 = self.h0_dense_layer(self.prenet_system(x, num_particle)[0, 0, :]).repeat(self.hp.rnn_layers, x.size(0), 1)

        x = x.permute(1, 0, 2)
        rnn_output, rnn_hn = self.rnn(x, rnn_h0)

        rnn_output = rnn_output.permute(1, 0, 2)
        postnet_output = self.postnet_dense_layer(rnn_output)
        postnet_output = self.postnet_dropout(postnet_output)

        y_ = self.postnet_restore(postnet_output, num_particle)  # (B, T_d, ReP + ImP)

        return y_, rnn_hn
