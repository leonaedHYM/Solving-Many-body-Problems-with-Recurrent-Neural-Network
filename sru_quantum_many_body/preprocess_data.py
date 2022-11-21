#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : preprocess_data.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 23.07.2019
# Last Modified Date: 06.08.2019
# Last Modified By  : zewangzhang <zewangzhang@tencent.com>
# -*- coding: utf-8 -*-
"""
Created on 2019/5/15 16:23

@author: Robin

Split data to train, val, test
Cut to required length
Separate real part and imaginary part
"""

import os
import glob
import shutil
import numpy as np

from hyperparams import hp


def split(raw_dir, train, val, test, systems=[2,3,4]):
    """
    Split the data in raw_dir to train, validation, test set in given proportion train:val:test
    :param raw_dir: raw data directory
    :param train: proportion of train-set
    :param val: proportion of validation-set
    :param test: proportion of test-set
    :return: None
    """
    total = train + val + test
    train_p, val_p, test_p = train / total, val / total, test / total

    if not os.listdir(raw_dir):
        print('no file in {}'.format(raw_dir))

    for system in systems:
        des_folder = str(system)
        d = os.path.join(raw_dir, 'N={}_time=1_step=0.002'.format(des_folder))
        print('handling {}'.format(des_folder))
        for p in (hp.val_data_dir, hp.test_data_dir, hp.train_data_dir):
            if not os.path.exists(os.path.join(p, des_folder)):
                os.makedirs(os.path.join(p, des_folder))

        # 文件数
        num = 0
        for _ in os.listdir(d):
            num += 1

        # 验证集,测试集,训练集
        count = 0
        for f in glob.glob('{}/*.npy'.format(d)):
            if count < int(num * val_p):
                shutil.copy(f, os.path.join(hp.val_data_dir, des_folder))
            elif int(num * val_p) <= count < int(num * (val_p + test_p)):
                shutil.copy(f, os.path.join(hp.test_data_dir, des_folder))
            else:
                shutil.copy(f, os.path.join(hp.train_data_dir, des_folder))
            count += 1

        print('finished spliting {}'.format(d))


def cut_separate(*data_dir, total_steps=None, systems=[2,3,4,5]):
    """
    将指定了路径的文件切割成需要的steps,并将实部虚部分开,即(n_tirals, enc_steps + dec_steps, 2 ** (N + 1))
    预处理后的数据的特征维度为2^(N+1)
    :param data_dir:
    :param total_steps:
    :return: None
    """
    for d in data_dir:
        for system in systems:
            folder = os.path.join(d, str(system))
            print('handling {}'.format(folder))
            for f in glob.glob('{}/*.npy'.format(os.path.join(d, folder))):

                arr = np.load(f)

                if total_steps:
                    arr = arr[:total_steps, :]

                if 'complex' in str(arr.dtype):
                    arr = np.concatenate((arr.real, arr.imag), axis=-1).astype(np.float)

                f_temp = f.rstrip('.npy') + '_temp.npy'
                np.save(f_temp, arr)
                os.remove(f)
                os.rename(f_temp, f)

            print('finished cutting and separating {}'.format(os.path.join(d, folder)))

if __name__ == '__main__':

    # Path
    systems = [8]
    for path in (hp.train_data_dir, hp.test_data_dir, hp.val_data_dir):
        if not os.path.exists(path):
            os.makedirs(path)

    print('start preprocess')

    split(hp.raw_data_dir, 10, 1, 1, systems=systems)
    print('split done')

    cut_separate(hp.train_data_dir, total_steps=None, systems=systems)
    cut_separate(hp.val_data_dir, hp.test_data_dir, total_steps=None, systems=systems)
    print('cut and separate done')

    print('preprocess finished')
