# -*- coding: utf-8 -*-
"""
Created on 2019/4/8

@author: Robin

helpful functions
"""
import os
import numpy as np


def get_num_particle(num_features):
    """
    从输入数据最后一个维度大小获得该batch是几粒子系统
    :param num_features: 输入数据最后一个维度大小
    :return: num_particle
    """
    convert_dict = {8: 2, 16: 3, 32: 4, 64: 5, 128: 6, 256: 7, 512: 8, 1024: 9, 2048: 10}
    assert num_features in convert_dict.keys()
    return convert_dict[num_features]


def get_num_params(model):
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_params])
    return num_params


def get_weights(model, save_dir, *layer, display_layers=False):
    """
    Save weight value in numpy file
    :param model: model that has loaded state dict
    :param save_dir: path to save weight files
    :param layer: key name that in the layer
    :param display_layers: if True, display all layers in the model first
    :return: None
    """
    for l, w in model.named_parameters():
        if display_layers:
            print(l, end='  ')
            print(w.shape)

        for ly in layer:
            if ly in l:
                arr = w.detach().numpy()
                np.save(os.path.join(save_dir, '{}.npy'.format(l)), arr)
                print('saved {}.'.format(l))
