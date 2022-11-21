# -*- coding: utf-8 -*-
"""
Created on 2019/4/8 13:55

@author: Robin

EarlyStopping
"""
import numpy as np


class EarlyStopping:
    def __init__(self, monitor='loss', min_delta=0, patience=0):
        """
        EarlyStopping
        :param monitor: quantity to be monitored. 'loss' or 'acc'
        :param min_delta: minimum change in the monitored quantity to qualify as an improvement
            i.e. an absolute change of less than min_delta, will count as no improvement.
        :param patience: number of epochs with no improvement after which training will be stopped
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self._wait = 0
        self._best = None
        self._best_epoch = None

        if 'loss' in self.monitor:
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1

    def judge(self, epoch, value):
        current = value

        if self._best is None:
            self._best = current
            self._best_epoch = epoch
            return

        if self.monitor_op(current - self.min_delta, self._best):
            self._best = current
            self._best_epoch = epoch
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                return True

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_loss(self):
        return self._best
