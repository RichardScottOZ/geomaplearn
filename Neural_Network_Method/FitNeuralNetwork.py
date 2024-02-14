# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:50:53 2023

@author: oakley

Detect fold areas and axes using UNET.

"""

import numpy as np
import tensorflow as tf
from UNET import UNET
from Generator import DataGenerator
import time

#Set some options.
use_topo = True
use_early_stopping = False
epochs = 250
use_XLA = False #This isn't working.
img_size = (256,256)
learning_rate = 1e-4 #Default is 1e-3
batch_size = 8#16
batches_per_epoch = int(800/batch_size)
validation_batches = int(200/batch_size)
nvalidation = validation_batches*batch_size
activation = 'relu' #relu or elu
first_dropout_layer = 2 #The first layer at which dropout should be applied. Indexing starts at 0. Layers on the decoder have the same numbers as the corresponding encoder ones.
dropout_prob = 0.0 #Set to 0 if you don't want to use dropout.
dropout_in_evaluation = True #If True, training is set to true when calling keras.layers.Dropout, so even in evaluation mode, dropout will be used.
normalization = None #Options are 'batch', 'group', 'layer', or None
optimizer = 'Adam' #Adam or RMSprop.
use_weights = False
weights = [1,1,2]
do_data_augmentation = False
fold_type = 'all' #'anticlines', 'synclines', or 'all'. Use 'all' even if only one type is labeled, if training data do not have separately named anticline or syncline files.
topo_scale = None
separable_convolutions = False
n_filters_start = 64 #Number of filters in the top UNET level.
n_steps = 5 #Number of levels to the UNET.

#Set up the data generators.
if use_topo:
    n_channels = 2
else:
    n_channels = 1
params1 = {'batch_size':batch_size,
         'dim':(256,256),
         'n_channels':n_channels,
         'shuffle':True,
         'area':True,
         'axis':True,
         'continuous_indices':True,
         'batches_per_epoch':batches_per_epoch,
         'augment':do_data_augmentation,
         'fold_type':fold_type,
         'topo_scale':topo_scale
         } #For training dataset
params2 = {'batch_size':batch_size,
         'dim':(256,256),
         'n_channels':n_channels,
         'shuffle':True,
         'area':True,
         'axis':True,
         'continuous_indices':False,
         'batches_per_epoch':validation_batches,
         'augment':False,
         'fold_type':fold_type,
         'topo_scale':topo_scale
         } #For validation and testing datasets.
if use_weights:
    params1['weights']=weights
    params2['weights']=weights
data_path = './RandomModels/Models'
ntrain = int(1e5) #Number of unique training models.
train_IDs = range(ntrain)
valid_IDs = range(ntrain,ntrain+nvalidation)
test_IDs = range(ntrain+200,ntrain+400)
train_generator = DataGenerator(path=data_path,
                                data_IDs=train_IDs,**params1)
valid_generator = DataGenerator(path=data_path,
                                data_IDs=valid_IDs,**params2)
test_generator = DataGenerator(path=data_path,
                                data_IDs=test_IDs,**params2)

# Build the model
num_classes = 3 #Off fold, in fold area, and on fold axis.
model = UNET(input_size=(img_size[0], img_size[1], n_channels), n_filters_start=n_filters_start, n_steps=n_steps, n_classes=num_classes, 
                     activation=activation, dropout_prob=dropout_prob, first_dropout_layer=first_dropout_layer, norm=normalization, 
                     separable=separable_convolutions, dropout_always=dropout_in_evaluation)
# model.summary()
if optimizer == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
elif optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
else:
    print('Error: Optimizer not recognized / not implemented.')
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", jit_compile=use_XLA)

#Create an early stopping callback.
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,)
if use_early_stopping:
    callbacks = [callback]
else:
    callbacks = []

#Train the model.
t1 = time.time()
history = model.fit(train_generator,validation_data=valid_generator, epochs=epochs,callbacks=callbacks,
                    steps_per_epoch=batches_per_epoch,validation_steps=validation_batches)
t2 = time.time()
print('Training took '+str(int(t2-t1))+' seconds')

#Save the model.
name = "DetectAreasAndAxisModel"
model.save(name,save_format="tf")
np.save(name+'_history.npy',history.history)

#Test the model on the test dataset.
model.evaluate(test_generator)
