#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : kld_evaluation.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 18.08.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt


def get_kl_divergence(p, q):
    """ 获取K-L散度
    """

    return scipy.stats.entropy(p, q)


def get_system_prob_dist(complexMatrix, Ns=5):
    """ Vector维度为 32

    """

    # 获取二进制表示
    real_part = complexMatrix[:, :(2**Ns)]
    imag_part = complexMatrix[:, (2**Ns):]

    form = '{0:0'+str(Ns)+'b}'
    binary_str = []
    for i in range(2**Ns):
        binary_str.append(str(form.format(i)))

    # probs = np.zeros((np.shape(real_part)[1], Ns))
    probs = np.zeros((np.shape(real_part)[0], 2**Ns))
    bin_probs = np.zeros((np.shape(real_part)[0], Ns))

    for t in range(np.shape(real_part)[0]):
        sum_abs = 0
        for i in range(len(real_part[0, :])):
            # 求概率和
            sum_abs += np.abs(np.complex(real_part[t, i],
                                         imag_part[t, i]))

        for i in range(len(real_part[0, :])):
            probs[t][i] = np.abs(np.complex(real_part[t, i],
                                            imag_part[t, i]))/sum_abs


    return probs, bin_probs


if __name__ == '__main__':
    data_path = "/zewangzhang/prediction"
    output_path = "/zewangzhang/kld_figures"

    #  particles = [2,3,4,5,6,7]
    particles = [8]
    for p in particles:
        files = glob.glob(os.path.join(data_path, 'system_{}_*.prediction.npy'.format(p)))

        KLD = []
        for f in files:
            if 'prediction' in f:
                fx = f
                fy = f.replace('prediction.npy', 'target.npy')
                #[100, 8]
                print(np.load(fx).shape)
                spd1, bin_spd1 = get_system_prob_dist(np.load(fx), Ns=p)
                print(np.shape(bin_spd1))
                spd2, bin_spd2 = get_system_prob_dist(np.load(fy), Ns=p)

                klds = []
                for pd1, pd2 in zip(spd1, spd2):
                    kld = get_kl_divergence(pd2, pd1)
                    klds.append(kld)
                KLD.append(klds)
        mean_kld = np.mean(KLD, axis=0)
        plt.plot(mean_kld, 'r')
        plt.savefig(os.path.join(output_path, 'system_{}.eps'.format(p)))
        plt.clf()

