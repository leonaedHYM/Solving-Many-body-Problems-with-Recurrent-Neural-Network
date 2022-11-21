#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_load.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 23.07.2019
# Last Modified Date: 26.07.2019
# Last Modified By  : zewangzhang <zewangzhang@tencent.com>
# -*- coding: utf-8 -*-
"""
Created on 2019/5/16

@author: Robin

Get batched data from different folders
The amount of data in different folders can be different
"""

import os
import numpy as np


def get_batch(data_dir, batch_size, permission, var_len=None, shuffle=True):
    # Get all files in data dir
    names = []
    for path, dirs, files in os.walk(data_dir):
        if files and int(path[-1]) in permission:
            subs = [os.path.join(path, file) for file in files]  # 指向某个系统的数据的路径列表
            names.append(np.array(subs))
    names = np.array(names)

    num_syst = len(names)  # number of systems
    num_samps = [len(name) for name in names]  # number of sample, available for different amount of data

    # Make indexs to get file at random
    indexs = []
    for syst in range(num_syst):
        idxs = [i for i in range(num_samps[syst])]
        if shuffle:
            np.random.shuffle(idxs)
        indexs.append(idxs)
    indexs = np.array(indexs)

    pockets = [i for i in range(num_syst)]  # 该epoch里还未用尽的系统
    pointers = [0] * num_syst  # 指示不同系统取到哪个位置的数据的指针列表

    while True:
        container = []
        syst = np.random.choice(pockets)  # 随机选择一个系统

        if pointers[syst] + batch_size >= num_samps[syst]:
            keys = indexs[syst][pointers[syst]: num_samps[syst]]
            files = names[syst][keys]
            for file in files:
                arr = np.load(file)
                if var_len:
                    arr[:np.random.randint(var_len), :] = 0
                container.append(arr)
            yield np.array(container)
            pockets.remove(syst)
            if not pockets:  # 所有系统用完,退出
                break

        else:
            keys = indexs[syst][pointers[syst]: pointers[syst] + batch_size]
            files = names[syst][keys]
            for file in files:
                arr = np.load(file)
                if var_len:
                    arr[:np.random.randint(var_len), :] = 0
                container.append(arr)
            yield np.array(container)
            pointers[syst] += batch_size


if __name__ == '__main__':
    from hyperparams import hp

    np.set_printoptions(threshold=np.nan)

    for data in get_batch(hp.train_data_dir, hp.batch_size, hp.permission, var_len=hp.enc_steps):
        print(data.shape, end='\t')

        # print(data[0,:,2])
