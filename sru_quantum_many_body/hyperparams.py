#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : hyperparams.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 22.07.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-
"""
Created on 2019/4/8

@author: Robin

HyperParameters

--- NOTICE ---
Train --- Check : finished_epoch, permission, batch_size, transformer.get_batch
Eval ---
"""


class Hyperparams:
    # Device
    device = 'cuda'  # 'cuda' or 'cpu'
    # prenet_feature,prenet_system,prenet_dense_layer,prenet_dropout,rnn,postnet_system,h0_dense_layer,postnet_dense_layer,postnet_dropout,postnet_restore
    exclude_module_name = ['prenet_dense_layer', 'rnn', 'h0_dense_layer', 'postnet_dense_layer'],

    # Path
    host_path = '/zewangzhang/project_longer'  # Path to save data and result

    data_dir = host_path + '/data'
    raw_data_dir = data_dir + '/raw'
    train_data_dir = data_dir + '/train'
    test_data_dir = data_dir + '/test'
    val_data_dir = data_dir + '/val'

    permission = (2,3,4,5,6,7)   # tuple,同时训练的粒子数系统,体现在get_batch()
    #permission = (7,)  # tuple,同时训练的粒子数系统,体现在get_batch()
    finished_epoch = 347  # int, should be same as ckpt_epoch_%d.pth file
    unit = 512
    layer_norm = True
    if len(permission) == 1:
        print('training single system')
        result_dir = host_path + '/result_single_7_512'
    elif unit == 256:
        result_dir = host_path + '/result'
    elif unit == 512:
        result_dir = host_path + '/result2'
    elif unit == 1024:
        result_dir = host_path + '/result3'
    else:
        result_dir = host_path + '/result4'
    if layer_norm is True:
        result_dir += '_layer_normed'
    ckpt_dir = result_dir + '/ckpt'
    event_dir = result_dir + '/event'  # Tensorboard event dir
    eval_dir = result_dir + '/eval'  # Evaluate files dir

    # Model (Attention)
    use_relu = False
    max_particle = 9  # 系统Embedding最大允许粒子数,10代表SystemEmbedding可以应付最多10粒子的系统

    feature_embedding_size = unit
    system_embedding_size = unit
    postnet_units = unit
    prenet_units = unit
    rnn_units = unit
    rnn_layers = 4

    # Switch
    add_system_encoding = True
    transfer = False  # transfer learning (迁移学习)
    var_len = None

    # Train
    # An argument in get_batch().
    num_epochs = 1000
    batch_size = 32  # (B, batch size) 64
    lr = 4e-6
    min_lr = 1e-6
    lr_decay_factor = 0.97  # with patience = 0 if no improvement in val_loss
    dropout_rate = 0.005

    # Early Stop
    patience = 20
    min_delta = 0

    # Eval
    eval_permission = (2,3,4,5,6,7)  # tuple,只能有一个值
    #  eval_permission = (7,)  # tuple,只能有一个值
    eval_epoch = 346  # int, should be same as ckpt_epoch_%d.pth file
    eval_batch_size = 1  # 有循环,所以要比较小
    initial_eval_steps = 1  # target seqence length (exclude enc_steps)
    total_eval_steps = 100  # target seqence length (exclude enc_steps)

hp = Hyperparams()
