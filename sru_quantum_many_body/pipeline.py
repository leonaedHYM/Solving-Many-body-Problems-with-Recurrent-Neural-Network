#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : pipeline.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 06.07.2019
# Last Modified Date: 07.07.2019
# Last Modified By  : zewangzhang <zewangzhang@tencent.com>
# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2019-01-14 11:02
# Email        : zzw922cn@gmail.com
# Filename     : pipeline.py
# Description  : data pipeline for fusion training of difference systems
# ******************************************************

import os
import toml
import numpy as np
import glob
from scaling_data import normalize_file


root_path = os.path.join(os.path.expanduser('~'), "many_body_system/rawdata/ed_lstm/")
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class DataPipeline(object):

    def __init__(self, config):
        self.idx = 0
        self.batch_size = config.batch_size
        self.root_path = config.data_path
        self.systems = config.systems
        self.num_step = config.num_step

    def get_batched_data(self, system, pad_to_system, mode='train'):
        """ 获取系统大小为system的数据"""

        data_path = os.path.join(self.root_path, "Ns={}/num_step={}".format(system,
                                                                            self.num_step))
        print(data_path)
        x_filenames = glob.glob(os.path.join(data_path, mode, 'x*'))
        count = 0
        x = []
        y = []

        for x_file in x_filenames:
            count += 1
            y_file = x_file.replace('/x', '/y')

            #x.append(normalize_file(x_file, system))
            #y.append(normalize_file(y_file, system))

            x.append(np.load(x_file))
            y.append(np.load(y_file))

            if count == self.batch_size:
                x = np.array(x, dtype=np.complex64)  #shape: [batch_size, sequence_length, 2**system]
                y = np.array(y, dtype=np.complex64)  #shape: [batch_size, sequence_length, 2**system] 

                # pad x and y into shape[batch_size, sequence_lengthm, 2**pad_to_system]
                x = np.pad(x, ((0,0), (0,0), (0, 2**pad_to_system-2**system)), 'constant')
                y = np.pad(y, ((0,0), (0,0), (0, 2**pad_to_system-2**system)), 'constant')

                #dataset.append([x, y, system])
                yield [x, y, system]
                count = 0
                x = []
                y = []


if __name__ == '__main__':
    configfile = "config.toml"
    config = Struct(**toml.load(configfile))
    dp = DataPipeline(config)
    gen = dp.get_batched_data(2, 8, 'train')
    for g in gen:
        print(g)
