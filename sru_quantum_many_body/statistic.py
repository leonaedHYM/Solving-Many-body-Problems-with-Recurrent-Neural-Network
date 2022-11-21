# -*- coding: utf-8 -*-

import numpy as np
import glob


def read_numpy(filename, Ns=5, keep_steps=500):
    data = np.load(filename).squeeze()[:keep_steps]
    prob = np.zeros((keep_steps, 2**Ns))
    for i in range(keep_steps):
        for j in range(2**Ns):
            prob[i][j] = data[i][j]**2+data[i][32+j]**2

    return np.sqrt(prob).T


if __name__ == '__main__':
    Ns = 5
    keep_steps = 500
    filenames = []
    data = []
    data_path = "/Users/zzw922cn/LSTM4QuantumManyBody/Ns={}/*.npy".format(Ns)
    files = glob.glob(data_path)
    total = []
    for filename in files:
        single = []
        if 'prediction' in filename:
            data1 = read_numpy(filename, Ns=Ns, keep_steps=keep_steps)
            data.append(data1)
            filename = filename.replace('prediction', 'target')
            data2 = read_numpy(filename, Ns=Ns, keep_steps=keep_steps)
            data.append(data2)
            data3 = np.abs(data1-data2)
            data.append(data3)
            print(data3.shape)
            for i in range(np.shape(data3)[1]):
                se = []
                for j in range(np.shape(data3)[0]):
                    se.append(data3[j][i])
                single.append(np.mean(se))
        total.append(single)
    print(np.mean(total, axis=0))
