# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:28:43 2017

@author: Quantum Liu
"""

import sys
import numpy as np
from vgg_fm import vgg_fm
from generators import read_data,reshape
from manager import GPUManager
from keras.callbacks import ModelCheckpoint
from callback import TargetStopping
if __name__=='__main__':
    gm=GPUManager()
    kwargs=dict(zip(['mode','version','batch_size'],sys.argv[1:]))
    mode,version,batch_size=list(map(lambda kd:kwargs.get(kd[0],kd[1]),zip(['mode','version','batch_size'],['vgg','v1',256])))
    batch_size=int(batch_size)
    model_name=mode+'_'+version
    with gm.auto_choice():
        (train_x,train_y),(test_x,test_y)=read_data('train'),read_data('test')
        train_x,test_x,train_y,test_y=reshape(train_x,False),reshape(test_x,False),np.expand_dims(train_y,-1),np.expand_dims(test_y,-1)
        input_shape=train_x.shape[1:]
        model=vgg_fm(input_shape)
        model.summary()
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
        model.fit(x=train_x,y=train_y,validation_data=(test_x,test_y),batch_size=batch_size,epochs=100,
                  callbacks=[TargetStopping(filepath=model_name+'.h5',monitor='val_acc',mode='max',target=0.94),
                             ModelCheckpoint(filepath=model_name+'.h5',save_best_only=True,monitor='val_acc')])
