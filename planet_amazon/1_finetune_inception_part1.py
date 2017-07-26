#
import numpy as np
import pandas as pd
from keras.applications import inception_v3
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import BatchNormalization, Dropout, Concatenate
from keras.models import Model
from keras import callbacks
from keras import losses, optimizers
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from helper_functions import helper_functions
from sklearn.metrics import fbeta_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import os
RANDOM_SEED = 43
np.random.seed(RANDOM_SEED)


# ** Load and Preprocess Data **
X_tr, y_tr = np.load('../../data/train/X_tr.npy'), np.load('../../data/train/y_tr.npy')
X_val, y_val = np.load('../../data/train/X_val.npy'), np.load('../../data/train/y_val.npy')
y_tr = y_tr.astype(np.float32)
y_val = y_val.astype(np.float32)
print 'X_tr.shape:', X_tr.shape
print 'y_tr.shape:', y_tr.shape
print 'X_val.shape:', X_val.shape
print 'y_val.shape:', y_val.shape
# # Merge
# X = np.concatenate([X_tr, X_val], axis=0)
# y = np.array(list(y_tr) + list(y_val))
# y = y.astype(np.float32)
# print 'X.shape:', X.shape
# print 'y.shape:', y.shape

#
X_val = inception_v3.preprocess_input(X_val.astype(np.float16))
print 'Preprocess done. (Only Validation)'


# Preprocess func
def pre_func(im):
    im = im.astype(np.float16)
    im = im / 255.0
    im = im - 0.5
    im = im * 2.0
    return im

##################################

# Constants
INPUT_SHAPE = (256, 256, 3)
N_CLASSES = 17

# ** Construct the Model **
# Base model from InceptionV3
base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
# Top Model (On top of base model)
x = base_model.output
x = GlobalAveragePooling2D(name='global_avg')(x)
x = Dropout(0.5)(x)
# fully-connected
x = Dense(1024, activation='relu', name='fc1')(x)
out_x = Dense(N_CLASSES, activation='sigmoid', name='predictions')(x)
# Make complete model
# (merge base model, top model)
model = Model(inputs=base_model.input, outputs=out_x)

# f_beta
def f_beta_k_(y_true, y_pred, t=0.1):
    return helper_functions.fbeta_k(y_true, y_pred, threshold=t)

##################################

# ** Top Model Train **
# Freeze layers (only open top)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
loss_ = losses.binary_crossentropy
optimizer_ = optimizers.Adam()
metrics_ = [f_beta_k_, 'accuracy']
model.compile(loss=loss_, optimizer=optimizer_, metrics=metrics_)

# Callbacks
m_q = 'val_loss'
model_path = './model_inception_1/model_inception_1.h5'
check_pt = callbacks.ModelCheckpoint(filepath=model_path, monitor=m_q, save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(patience=2, monitor=m_q, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(patience=0, factor=0.33, monitor=m_q, verbose=1)
callback_list = [check_pt, reduce_lr, early_stop]

# Data Generator
datagen = ImageDataGenerator(vertical_flip=True,                              
                             horizontal_flip=True,  
                             rotation_range=180,
                             fill_mode='reflect',                          
                             preprocessing_function=pre_func)

# Batch Size
batch_size = 64

# Fit
model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=batch_size),
                    validation_data=(X_val, y_val),  
                    epochs=3,
                    steps_per_epoch=len(X_tr)/batch_size,                    
                    callbacks=callback_list)  
print 'Top Model Train Done.'