# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from kld_evaluation import get_system_prob_dist


def plot_spin_direction_function(coefficient, name):

    ax = plt.gca()
    im = ax.imshow(coefficient, cmap='hot',
                   aspect='auto', vmin=0, vmax=1)
    plt.xlabel('t')
    plt.ylabel('coefficient')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax, format='%.0e')
    ax.set_title(name)
    plt.subplots_adjust(hspace=0, wspace=0)
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
    filename = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns='+str(Ns) \
        + '/target_100' +'.npy'
    _, spin_direction_5 = get_system_prob_dist(np.load(filename), Ns=Ns)
    print(spin_direction_5)
    plot_spin_direction_function(spin_direction_5.T, '5t')

    Ns = 6
    filename = '/Users/zzw922cn/LSTM4QuantumManyBody/Ns='+str(Ns) \
        + '/target_100'+'.npy'
    _, spin_direction_6 = get_system_prob_dist(np.load(filename), Ns=Ns)
    plot_spin_direction_function(spin_direction_6.T, '6t')
