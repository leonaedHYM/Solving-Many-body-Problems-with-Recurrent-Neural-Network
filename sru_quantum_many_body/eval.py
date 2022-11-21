#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : eval.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 27.07.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-

import os
import math
import time

import numpy as np
import torch

from torch_model import RNNModel
from lib import get_batch, get_num_particle
from hyperparams import hp
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main():
    # Load Model
    model = RNNModel(hp, inference=True)
    model.load_state_dict(torch.load(hp.ckpt_dir + '/ckpt_epoch_{:02d}.pth'.format(hp.eval_epoch)))
    print('Successfully load ckpt_epoch_{:02d}.pth'.format(hp.eval_epoch))
    print('Start evaluating')
    model.eval()
    model.to(hp.device)

    count = 0
    targets, predictions = [], []  # 用于将所有batch合并
    consumes = {'2':[], '3':[], '4':[], '5':[], '6':[], '7':[]}
    # Evaluate
    for data in get_batch(hp.test_data_dir, hp.eval_batch_size, hp.eval_permission, shuffle=False):

        preds = data[:, :hp.initial_eval_steps, :]  # 记录一个batch里的预测值  (batch, enc_steps, num_units)

        x_batch = torch.Tensor(data[:, :hp.initial_eval_steps, :]).to(hp.device)  # (batch, enc_steps, num_units)
        num_particle = get_num_particle(x_batch.size(-1))
        pred = torch.zeros(x_batch.size(0), hp.initial_eval_steps+hp.total_eval_steps, x_batch.size(-1)).to(
            hp.device)  # (batch, dec_steps, num_units)
        pred[:, :hp.initial_eval_steps, :] = x_batch

        inputx = x_batch
        t1 = time.time()
        rnn_hn = None
        for i in range(hp.total_eval_steps):  # 未必要是dec_steps(50),可以减小
            # 老版本，比较耗时
            _, _pred = model(inputx, inputx, num_particle)  # (batch, dec_steps, num_units)
            pred[:, hp.initial_eval_steps+i, :] = _pred[:, -1, :]
            inputx = torch.cat([inputx, _pred[:, -1:, :]], 1)

            # 新版本，速度较快
            #  _pred, rnn_hn = model.generate(inputx, num_particle, rnn_hn)
            #  pred[:, hp.initial_eval_steps+i, :] = _pred[:, -1, :]
            #  inputx = _pred
        t2 = time.time()
        count += 1
        if count > 2:
            consumes[str(num_particle)].append(t2-t1)

        preds = np.concatenate((preds, pred.to('cpu').detach().numpy()), axis=1)  # (batch, enc_steps + dec_steps, n..)

        known_loss = ((preds[:, :hp.initial_eval_steps+hp.total_eval_steps, :] - data[:, :hp.initial_eval_steps+hp.total_eval_steps, :]) ** 2).mean(axis=1)
        print(known_loss)
        known_loss = known_loss.mean()
        for i in range(hp.eval_batch_size):
            #  for j in range(2**num_particle):
                #  if (i == 0 and num_particle == 2 and j ==2 and count == 51) or (i == 0 and num_particle == 3 and j == 7 and count == 42) or (i == 0 and num_particle == 4 and j == 15 and count == 20) or (i == 0 and num_particle == 5 and j == 2 and count == 23) or (i == 0 and num_particle == 6 and j == 4 and count == 10) or (i == 0 and num_particle == 7 and j == 42 and count == 22):
                #  plt.plot(data[i,:hp.total_eval_steps,j])
                #  plt.plot(preds[i,:hp.total_eval_steps,j])
                #  plt.savefig('test_{}_{}_{}_sample_{}.eps'.format(i, num_particle, j, count))
                #  plt.clf()
            print(np.array(preds[i,:hp.total_eval_steps]).shape)
            np.save('/zewangzhang/prediction/system_{}_{}_{}.target.npy'.format(num_particle, i, count), data[i,:hp.total_eval_steps])
            np.save('/zewangzhang/prediction/system_{}_{}_{}.prediction.npy'.format(num_particle, i, count), preds[i,:hp.total_eval_steps])
        print('Current batch: {}, System: {}, projection steps: {}, known loss: {}'.format(count, num_particle, hp.total_eval_steps, known_loss))
        print('Particle:', num_particle, '耗时:', t2-t1)
    for k,v in consumes.items():
        v.pop(v.index(max(v)))
        v.pop(v.index(min(v)))
        print(k, np.mean(v))


if __name__ == '__main__':

    # Path
    if not os.path.exists(hp.eval_dir):
        os.makedirs(hp.eval_dir)

    main()
