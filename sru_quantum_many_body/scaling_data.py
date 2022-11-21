#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : scaling_data.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 30.06.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
""" 对复数实部和虚部分别进行均值方差归一化"""

import os
import glob
import numpy as np
from numpy.linalg import matrix_rank
import argparse


data_path = os.path.join(os.path.expanduser('~'), 'many_body_system/rawdata/ed_lstm/')

def normalize_arr(a, mean, std):
    """ 归一化"""

    result = []
    for aa in a:
        r = (aa-mean)/std
        result.append(r)
    return np.array(result, dtype=np.float32)


def denormalize_arr(a, mean, std):
    """ 反归一化"""

    result = []
    for aa in a:
        r = aa*std + mean
        result.append(r)
    return np.array(result, dtype=np.float32)

def get_mean_variance_coefficient(input_root_dir, output_dir, num_particles=[2,3,4,5]):
    """ 获取全局均值和全局方差系数 """

    count = 0
    for p in num_particles:
        total = []
        for entry in glob.glob(os.path.join(input_root_dir, str(p), '*.npy')):
            count += 1
            array = np.load(entry)
            for ca in array:
                total.append(ca)
        total = np.array(total, dtype=np.float32)
        mean_vec = np.mean(total, 0)
        std_vec = np.std(total, 0)
        mean_f = os.path.join(output_dir, 'mean_vec_{}'.format(p))
        std_f = os.path.join(output_dir, 'std_vec_{}'.format(p))
        np.save(mean_f, mean_vec)
        np.save(std_f, std_vec)

def generate_normalized_files(input_root_dir, mean_std_dir, output_dir, num_particle=2):
    """ 生成归一化后的文件 """

    os.makedirs(os.path.join(output_dir, '{}'.format(num_particle)), exist_ok=True)
    for entry in glob.glob(os.path.join(input_root_dir, str(num_particle), '*.npy')):
        normalized_array = normalize_file(entry, num_particle, mean_std_dir)
        normalized_f = os.path.join(output_dir, '{}'.format(num_particle), os.path.basename(entry))
        np.save(normalized_f, normalized_array)

def check_whether_normalized(raw_dir, normalized_dir, mean_std_dir, num_particle):
    """ 测试原始数据文件夹与归一化后的文件夹数据是否正确 """
    for entry in glob.glob(os.path.join(raw_dir, str(num_particle), '*.npy')):
        bn = os.path.basename(entry)
        normalized_f = os.path.join(normalized_dir, str(num_particle), bn)
        raw_arr = np.load(entry)
        denormalized_arr = denormalize_file(normalized_f, num_particle, mean_std_dir)
        print(np.array_equal(raw_arr, denormalized_arr))
        print(raw_arr)
        print(np.load(normalized_f))
        print(denormalized_arr)

def normalize_file(filename, particle, mean_std_dir):
    """ 归一化文件"""
    raw_array = np.load(filename)
    mean = np.load(os.path.join(mean_std_dir, 'mean_vec_{}.npy'.format(particle)))
    std = np.load(os.path.join(mean_std_dir, 'std_vec_{}.npy'.format(particle)))
    normalized_array = normalize_arr(raw_array, mean, std)
    return normalized_array

def denormalize_file(filename, particle, mean_std_dir):
    """ 反归一化文件"""
    raw_array = np.load(filename)
    mean = np.load(os.path.join(mean_std_dir, 'mean_vec_{}.npy'.format(particle)))
    std = np.load(os.path.join(mean_std_dir, 'std_vec_{}.npy'.format(particle)))
    denormalized_array = denormalize_arr(raw_array, mean, std)
    return denormalized_array


def unit_test(filename, mean_std_dir, particle):
    """ 归一化与反归一化测试代码 """

    raw_array = np.load(filename)
    print(raw_array[:1])
    mean = np.load(os.path.join(mean_std_dir, 'mean_vec_{}.npy'.format(particle)))
    std = np.load(os.path.join(mean_std_dir, 'std_vec_{}.npy'.format(particle)))
    normalized_array = normalize_arr(raw_array, mean, std)
    print(normalized_array[:1])
    denormalized_array = denormalize_arr(normalized_array, mean, std)
    print(denormalized_array[:1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", help="the number of particles", type=int)
    args = parser.parse_args()
    input_root_dir='/zewangzhang/project_longer/data/test'
    output_dir='/zewangzhang/project_longer/data_normalized/test'
    mean_std_dir = '/zewangzhang/mean_std_files'
    os.makedirs(mean_std_dir, exist_ok=True)
    Ns = args.ns
    get_mean_variance_coefficient(input_root_dir, mean_std_dir, num_particles=[Ns])
    generate_normalized_files(input_root_dir, mean_std_dir, output_dir, num_particle=Ns)
    #check_whether_normalized(input_root_dir, output_dir, mean_std_dir, Ns)
    #unit_test(os.path.join(input_root_dir, '{}/x_9.npy'.format(Ns)), mean_std_dir, Ns)
