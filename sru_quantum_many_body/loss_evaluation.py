# -*- coding: utf-8 """

import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def smooth(scalars, weight):
    """
    平滑化曲线
    """

    last = scalars[0]  
    smoothed = list()
    count = 0
    for point in scalars:
        count += 1
        smoothed_val = last * weight + (1 - weight) * point  
        if count%10 == 0:
            smoothed.append(smoothed_val)
        last = smoothed_val                                  

    return smoothed

def plot_loss_from_csv(csv_file, title):
    """ 从CSV文件中读取数据并可视化
    """

    data = pd.read_csv(csv_file, delimiter=',')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.plot(smooth(data['Value'], 0.98), 'b', marker='*')
    plt.title(title)
    plt.savefig(title+'.eps')

if __name__ == '__main__':

    files = glob.glob('/Users/zzw922cn/LSTM4QuantumManyBody/csv/*/*.csv')
    for f in files:
        plt.clf()
        part1 = f.split('/')[-2]
        part2 = f.split('-')[-1].split('.')[0]
        title = '_'.join([part1, part2])
        plot_loss_from_csv(f, title)
    
