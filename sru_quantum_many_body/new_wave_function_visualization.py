#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : new_wave_function_visualization.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 30.08.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt


def plot_wave_function(coefficients, size):
    # tab10,tab20_r,Accent_r,Paired_r,Paired

    fig, axes = plt.subplots(nrows=len(coefficients), ncols=1)
    cm = 'Blues'
    #  cm = 'Purples'
    for coefficient, ax in zip(coefficients, axes.flat):
        im = ax.imshow(coefficient, vmin=np.min(coefficient), vmax=np.max(coefficient),
                       cmap=cm, aspect='auto')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig("./{}_{}.eps".format(cm, size))


def read_numpy(filename, Ns, keep_steps):
    data = np.load(filename).squeeze()[:keep_steps]
    prob = np.zeros((keep_steps, 2**Ns))
    for i in range(keep_steps):
        for j in range(2**Ns):
            prob[i][j] = np.sqrt(data[i][j]**2+data[i][2**Ns+j]**2)

    return np.array(prob).T


if __name__ == '__main__':
    idx = [3298, 3095, 3056, 3336, 3374, 4322]
    particles = [2,3,4,5,6,7]

    keep_steps = 100
    filenames = []
    data = []

    root_generation_path = '/zewangzhang/prediction'

    for Ns, idd in zip(particles, idx):
        data = []
        count = 0
        filename = os.path.join(root_generation_path, 'system_{}_0_{}.prediction.npy'.format(Ns, idd))

        wave1 = read_numpy(filename, Ns=Ns, keep_steps=keep_steps)
        data.append(wave1)
        filename = filename.replace('prediction.npy', 'target.npy')
        wave2 = read_numpy(filename, Ns=Ns, keep_steps=keep_steps)
        data.append(wave2)
        diff = abs(wave1-wave2)
        print(wave1[0])
        print(wave2[0])
        #  data.append(diff)


        #  for i in range(np.shape(diff)[1]):
        #      se = []
        #      ratio = []
        #      for j in range(np.shape(diff)[0]):
        #          se.append(diff[j][i])
        #          ratio.append(diff[j][i]/wave2[j][i])
        #      print(i, np.mean(se), np.mean(ratio))

        plot_wave_function(data, size=Ns)
