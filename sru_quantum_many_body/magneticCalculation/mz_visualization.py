#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mz_visualization.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 19.08.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding:utf-8 -*-
import numpy as np
import glob
import argparse
from EvoED import getMz
import os
from matplotlib import pyplot as plt


def read_numpy(filename, Ns, keep_steps):
    data = np.load(filename).squeeze()[:keep_steps]
    wave = np.zeros((keep_steps, 2**Ns)).astype(np.complex64)
    for i in range(keep_steps):
        for j in range(2**Ns):
            wave[i][j] = complex(data[i][j], data[i][2**Ns+j])

    return wave


if __name__ == '__main__':
    idx = [3298, 3095, 3056, 3336, 3374, 4322]
    particles = [2,3,4,5,6,7]

    keep_steps = 100
    filenames = []
    data = []

    root_generation_path = '/zewangzhang/prediction'

    for Ns in particles:
        count = 0
        filenames = glob.glob(os.path.join(root_generation_path, 'system_{}_*.prediction.npy'.format(Ns)))
        for filename in filenames:
            count += 1
            wave1 = read_numpy(filename, Ns=Ns, keep_steps=keep_steps)
            filename = filename.replace('prediction.npy', 'target.npy')
            wave2 = read_numpy(filename, Ns=Ns, keep_steps=keep_steps)
            predicted_mzs = []
            target_mzs = []
            for w1, w2 in zip(wave1, wave2):
                mz1 = getMz(Ns, w1)
                mz2 = getMz(Ns, w2)
                predicted_mzs.append(mz1)
                target_mzs.append(mz2)
            predicted_mzs = np.array(predicted_mzs).T
            target_mzs = np.array(target_mzs).T

            plt.title("magnetic Z by ED and LSTM based method")
            for iid in range(Ns):
                plt.subplot(Ns, 1, iid+1)
                plt.plot(predicted_mzs[iid], 'r')
                plt.plot(target_mzs[iid], 'g')
            plt.savefig("/zewangzhang/magnetic_figures/{}_{}.eps".format(Ns, count))
            print(" save into /zewangzhang/magnetic_figures/{}_{}.eps".format(Ns, count))
            plt.clf()
