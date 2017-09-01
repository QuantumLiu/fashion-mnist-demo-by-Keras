# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 13:23:20 2017

@author: Quantum Liu
"""
import warnings
import numpy as np
from keras.callbacks import Callback

class TargetStopping(Callback):
    '''
    early stopping by target
    '''
    def __init__(self,filepath='',monitor='val_acc',target=0.99,mode='max',patience=0):
        self.monitor=monitor
        self.target=target
        self.patience=patience
        self.wait=0
        self.early_stopped=False
        self.filepath=filepath
        self.stopped_epoch=0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
                
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        if self.monitor_op(current,self.target):
            self.wait+=1
            if self.wait>=self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.early_stopped=True
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: target stopping' % (self.stopped_epoch))
        if self.early_stopped and self.filepath:
            self.model.save(self.filepath,overwrite=True)