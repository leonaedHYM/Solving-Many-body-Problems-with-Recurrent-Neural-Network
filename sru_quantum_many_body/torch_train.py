#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : torch_train.py
# Author            : zewangzhang <zewangzhang@tencent.com>
# Date              : 22.07.2019
# Last Modified Date: 04.10.2019
# Last Modified By  : zewangzhang <zzw922cn@gmail.com>
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import logging
from datetime import datetime as dt

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from hyperparams import hp
from lib import EarlyStopping, get_batch, get_num_particle, get_num_params
from torch_model import RNNModel

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def gen_mask():
    np.random.binomial(size=3, n=1, p= 0.5)


def main():
    writer = SummaryWriter(hp.event_dir)
    earlystop = EarlyStopping(monitor='loss', min_delta=hp.min_delta, patience=hp.patience)

    # Model
    model = RNNModel(hp)

    count = 0
    # for name, module in model.named_children():
    #     count += 1
    #     print(name)
    #     print(hp.exclude_module_name)

    #     if name in hp.exclude_module_name[0]:
    #         for param in module.parameters():
    #             param.requires_grad = False
    #             print('skipping...', param)

    # Number of trainable parameters
    logging.info('Build model.\nNumber of trainable parameters: {}'.format(get_num_params(model)))
    model.to(hp.device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=hp.lr_decay_factor, verbose=True,
                                                    patience=0, min_lr=hp.min_lr)

    # Load existed weights
    if hp.finished_epoch and os.path.exists(os.path.join(hp.result_dir, 'history.pkl')):
        with open(os.path.join(hp.result_dir, 'history.pkl'), 'rb') as fin:
            history = pickle.load(fin)
        logging.info('Successfully load history.')
    else:
        history = {'total_finished_batches': 0}
        logging.info('No train history found.')
    t_f_batches = history['total_finished_batches']  # total finished batches will be a global steps in tensorboard

    if hp.finished_epoch and os.path.exists(os.path.join(hp.ckpt_dir, 'optimizer.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(hp.ckpt_dir, 'optimizer.pth')))
        logging.info('Successfully load optimizer state.')

    if hp.finished_epoch and os.path.exists(
            os.path.join(hp.ckpt_dir, 'ckpt_epoch_{:02d}.pth'.format(hp.finished_epoch))):
        model.load_state_dict(torch.load(os.path.join(hp.ckpt_dir, 'ckpt_epoch_{:02d}.pth'.format(hp.finished_epoch))))
        logging.info('Successfully load model state.')

    # Transfer learning
    if hp.transfer and hp.finished_epoch:
        for name, param in model.named_parameters():
            if 'lookup_table' not in name:
                param.requires_grad = False
        logging.info('Transfer learning mode: ON.')
        logging.info('Transfer to {} particles system.'.format(hp.permission[0]))
        logging.info('Number of trainable parameters: {}'.format(get_num_params(model)))

    start_epoch = int(hp.finished_epoch + 1) if hp.finished_epoch else 1
    batch_per_epoch = len(list(get_batch(hp.train_data_dir, hp.batch_size, permission=hp.permission)))

    # Train
    logging.info('Start training.')
    for epoch in range(start_epoch, hp.num_epochs + 1):
        model.train()
        cur_batch = 0
        time_start = dt.now()
        for data in get_batch(hp.train_data_dir, batch_size=hp.batch_size, permission=hp.permission,
                              var_len=hp.var_len):
            x_batch = torch.Tensor(data[:, :-1, :]).to(hp.device)
            y_batch = torch.Tensor(data[:, 1:, :]).to(hp.device)
            num_particle = get_num_particle(x_batch.size(-1))

            optimizer.zero_grad()
            loss, _ = model(x_batch, y_batch, num_particle)
            loss.backward()
            optimizer.step()

            t_f_batches += 1
            cur_batch += 1

            if cur_batch % 10 == 0:
                msg = 'epoch: {}, system: {}, batch: {}/{}, train_loss: {}'
                print(msg.format(epoch, num_particle, cur_batch, batch_per_epoch, loss.item()))
            if t_f_batches % 10 == 0:
                writer.add_scalar('train_loss', loss.cpu().item(), t_f_batches)
            if cur_batch % 20 == 0:
                with open(os.path.join(hp.result_dir, 'all_scalars.csv'), 'a', encoding='utf-8') as fout:
                    msg = '{},{},{}\n'.format(epoch, cur_batch, loss.item())
                    fout.write(msg)

        time_end = dt.now()

        # Save global step (total finished batches)
        history['total_finished_batches'] = t_f_batches
        with open(os.path.join(hp.result_dir, 'history.pkl'), 'wb') as fout:
            pickle.dump(history, fout)

        torch.save(model.state_dict(), os.path.join(hp.ckpt_dir, 'ckpt_epoch_{:02d}.pth'.format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(hp.ckpt_dir, 'optimizer.pth'))
        # 删除 (hp.patience+1) 个epoch前的ckpt
        if os.path.exists(os.path.join(hp.ckpt_dir, 'ckpt_epoch_{:02d}.pth'.format(epoch - hp.patience - 1))):
            os.remove(os.path.join(hp.ckpt_dir, 'ckpt_epoch_{:02d}.pth'.format(epoch - hp.patience - 1)))

        logging.info('Finish epoch {}, time using: {}, ckpt have been saved.'.format(epoch, time_end - time_start))

        # Early Stopping
        model.eval()
        val_losses = []
        particle_val_losses = {'2':[],'3':[],'4':[],'5':[],'6':[],'7':[]}
        for data in get_batch(hp.val_data_dir, batch_size=hp.batch_size // 4, permission=hp.eval_permission,
                              var_len=hp.var_len):
            x_batch = torch.Tensor(data[:, :-1, :]).to(hp.device)
            y_batch = torch.Tensor(data[:, 1:, :]).to(hp.device)
            num_particle = get_num_particle(x_batch.size(-1))

            val_loss, _ = model(x_batch, y_batch, num_particle)
            val_losses.append(val_loss.item())
            particle_val_losses[str(num_particle)].append(val_loss.item())

        val_loss_mean = np.mean(val_losses)
        lr_decay.step(val_loss_mean)  # learning rate decay
        logging.info('val_loss: {}'.format(val_loss_mean))
        for k,v in particle_val_losses.items():
            logging.info('spin:{}, GTA loss:{}'.format(k, np.mean(v)))
        writer.add_scalar('val_loss', val_loss_mean, epoch)

        if earlystop.judge(epoch, val_loss_mean):
            logging.info('Early stop at epoch {}, with loss {}'.format(epoch, val_loss_mean))
            logging.info('Best perform epoch: {}, with loss {}'.format(earlystop.best_epoch, earlystop.best_loss))
            break

    logging.info('Done')


if __name__ == '__main__':
    # Path
    os.makedirs(hp.result_dir, exist_ok=True)
    os.makedirs(hp.ckpt_dir, exist_ok=True)
    os.makedirs(hp.event_dir, exist_ok=True)

    # Logging Setting
    logging.basicConfig(
        filename=hp.result_dir + '/train.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M',
        level=logging.INFO
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M'))
    logging.getLogger('').addHandler(console)

    main()
