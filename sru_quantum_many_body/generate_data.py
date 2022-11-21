#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : generate_data.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 22.07.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-

import os
import time
import argparse
import glob

import numpy as np

from data_generator.ED import get_psi_seq


def make_data(dirpathname, ns, n_trial, time, step, Zg, Zh):
    print('Generating Data...')

    for f in glob.glob('{}/*.npy'.format(dirpathname)):
        os.remove(f)

    # from 10000 starting
    for i in range(n_trial):
        filename = os.path.join(dirpathname, 'x_{}'.format(i + 1))

        print('Generating {} data {}/{}'.format(ns, i + 1, n_trial))

        _seq = get_psi_seq(time, step, ns, Zg, Zh)
        np.save(filename, np.array(_seq, dtype=np.complex64))
        print('x data, shape is {}, saved to {}'.format(np.shape(_seq), filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", help="the number of particles", type=int)
    parser.add_argument("--root_save_path", help="the path of generated data", default='/zewangzhang/project_longer/data/raw', type=str)
    parser.add_argument('--n_trial', help='number of data', default=5000, type=int)
    parser.add_argument('--time', help='time, default to 1', default=0.2, type=float)
    parser.add_argument('--step', help='step, default to 0.002', default=0.002, type=float)

    args = parser.parse_args()

    if not os.path.isdir(args.root_save_path):
        os.makedirs(args.root_save_path)

    dirpathname = os.path.join(args.root_save_path, 'N={}_time={}_step={}'.format(args.ns, args.time, args.step))
    if not os.path.isdir(dirpathname):
        os.makedirs(dirpathname)

    Zg = -1.05 # default: -1.05
    Zh = 0.5 # default: 0.5

    t1 = time.time()
    make_data(dirpathname, args.ns, args.n_trial, args.time, args.step, Zg=Zg, Zh=Zh)
    t2 = time.time()
    print('耗时:', t2-t1)
