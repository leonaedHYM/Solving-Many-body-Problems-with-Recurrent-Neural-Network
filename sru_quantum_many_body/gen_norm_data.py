#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gen_norm_data.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 15.07.2019
# Last Modified Date: 15.07.2019
# Last Modified By  : zewangzhang <zewangzhang@tencent.com>

import numpy as np
import glob
import os
from scaling_data import normalize_complex_arr
from scaling_data import denormalize_complex_arr

def gen_norm_data(stat_dir, output_dir, systems=[2,3,4,5]):
    """ 生成归一化后的数据 """
    for Ns in systems:
        real_mean = np.load(os.path.join(stat_dir, 'real_mean_vec_{}.npy'.format(Ns)))
        real_std = np.load(os.path.join(stat_dir, 'real_std_vec_{}.npy'.format(Ns)))
        imag_mean = np.load(os.path.join(stat_dir, 'imag_mean_vec_{}.npy'.format(Ns)))
        imag_std = np.load(os.path.join(stat_dir, 'imag_std_vec_{}.npy'.format(Ns)))
        files = glob.glob(os.path.join(os.path.expanduser('~'), 'many_body_system/rawdata/ed_lstm/Ns={}/num_step=500/*/*.npy'.format(Ns)))
        for fn in files:
            print('processing {}'.format(fn))
            raw_array = np.load(fn)
            new_fn = os.path.join(output_dir, fn.split('ed_lstm/')[1])
            print(new_fn)
            os.makedirs(os.path.dirname(new_fn), exist_ok=True)
            normalized_array = normalize_complex_arr(raw_array, real_mean, real_std, imag_mean, imag_std)
            np.save(new_fn, normalized_array)
            print('finish {}'.format(new_fn))

def gen_denorm_data(stat_dir, output_dir, systems=[2,3,4,5]):
    """ 生成归一化后的数据 """
    for Ns in systems:
        real_mean = np.load(os.path.join(stat_dir, 'real_mean_vec_{}.npy'.format(Ns)))
        real_std = np.load(os.path.join(stat_dir, 'real_std_vec_{}.npy'.format(Ns)))
        imag_mean = np.load(os.path.join(stat_dir, 'imag_mean_vec_{}.npy'.format(Ns)))
        imag_std = np.load(os.path.join(stat_dir, 'imag_std_vec_{}.npy'.format(Ns)))
        files = glob.glob(os.path.join(os.path.expanduser('~'), 'many_body_system/rawdata/ed_lstm_norm/Ns={}/num_step=500/*/*.npy'.format(Ns)))
        for fn in files:
            print('processing {}'.format(fn))
            raw_array = np.load(fn)
            new_fn = os.path.join(output_dir, fn.split('ed_lstm_norm/')[1])
            print(new_fn)
            os.makedirs(os.path.dirname(new_fn), exist_ok=True)
            denormalized_array = denormalize_complex_arr(raw_array, real_mean, real_std, imag_mean, imag_std)
            np.save(new_fn, denormalized_array)
            print('finish {}'.format(new_fn))


if __name__ == '__main__':
    stat_dir = os.path.join(os.path.expanduser('~'), 'many_body_system/mean_std_files')
    output_dir = os.path.join(os.path.expanduser('~'), 'many_body_system/rawdata/ed_lstm_norm/')
    gen_norm_data(stat_dir, output_dir, systems=[2,3,4,5])
    # for validation
    # output_dir = os.path.join(os.path.expanduser('~'), 'many_body_system/rawdata/ed_lstm_denorm/')
    # gen_denorm_data(stat_dir, output_dir, systems=[2,3,4,5])
