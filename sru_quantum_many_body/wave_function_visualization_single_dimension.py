# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


def plot_wave_function_single(prediction, target, name):

    ax = plt.gca()
    idx = 5
    plt.plot(prediction[idx, :], 'r')
    plt.plot(target[idx, :], 'g')
    plt.xlabel('t')
    plt.ylabel('one dimension of coefficient')
    ax.set_title(name)
    plt.savefig(name+'.eps')
    plt.clf()


def read_numpy(filename, Ns=5, keep_steps=500):
    data = np.load(filename).squeeze()[:keep_steps]
    prob = np.zeros((keep_steps, 2**Ns))
    for i in range(keep_steps):
        for j in range(2**Ns):
            prob[i][j] = data[i][j]**2+data[i][32+j]**2

    return np.array(prob).T


if __name__ == '__main__':
    Ns = 5
    keep_steps = 500
    idx = 500
    filename = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns='+str(Ns) \
        + '/prediction_{}.npy'.format(idx)
    data1 = read_numpy(filename, Ns=Ns)
    filename = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns=' + str(Ns) + \
        '/target_{}.npy'.format(idx)
    data2 = read_numpy(filename, Ns=Ns)
    plot_wave_function_single(data1, data2, name=str(Ns)+'_' +
                              filename.split('/')[-1].split('_')[0]+'_' +
                              str(keep_steps)+'_single')
    Ns = 6
    filename = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns='+str(Ns) \
        + '/prediction_{}.npy'.format(idx)
    data1 = read_numpy(filename, Ns=Ns)
    filename = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns=' + str(Ns) + \
        '/target_{}.npy'.format(idx)
    data2 = read_numpy(filename, Ns=Ns)
    plot_wave_function_single(data1, data2, name=str(Ns)+'_' +
                              filename.split('/')[-1].split('_')[0]+'_' +
                              str(keep_steps)+'_single')
