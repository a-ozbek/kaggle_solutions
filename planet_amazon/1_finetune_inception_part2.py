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
# f_beta
def f_beta_k_(y_true, y_pred, t=0.1):
    return helper_functions.fbeta_k(y_true, y_pred, threshold=t)

# Load the model
model_path = './model_inception_1/model_inception_1.h5'
model = load_model(model_path, custom_objects={'f_beta_k_':f_beta_k_})
print 'Model is loaded.'
##################################

# ** Finetune **  
# Epochs, Learning Rate
n_epochs = [10, 13, 12, 3]
lr_rates = [0.0100, 0.0050, 0.0010, 0.0001]

# Callbacks
# learning rate scheduler function
def lr_schedule_func(n_epoch):
    n_epochs = [10, 13, 12, 3]
    lr_rates = [0.0100, 0.0050, 0.0010, 0.0001]
    def make_lr_arr(n_epochs, lr_rates):
        lr_arr = []
        for n, lr in zip(n_epochs, lr_rates):
            lr_arr += n * [lr]
        return lr_arr
    lr_arr = make_lr_arr(n_epochs, lr_rates)
    return lr_arr[n_epoch]
m_q = 'val_loss'
check_pt = callbacks.ModelCheckpoint(filepath='./model_inception_1/model_inception_1.h5', monitor=m_q, save_best_only=True, verbose=1)
tb_board = callbacks.TensorBoard(log_dir='./model_inception_1/tb_logs/', write_graph=False)
csv_logger = callbacks.CSVLogger(filename='./model_inception_1/log.csv', append=True)
lr_schedule = callbacks.LearningRateScheduler(lr_schedule_func)
reduce_lr = callbacks.ReduceLROnPlateau(patience=1, factor=0.5, monitor=m_q, verbose=1)
early_stop = callbacks.EarlyStopping(patience=3, monitor=m_q, verbose=1)

callback_list = [check_pt, tb_board, csv_logger, reduce_lr, early_stop]

# Data Generator
datagen = ImageDataGenerator(vertical_flip=True,                              
                             horizontal_flip=True,  
                             rotation_range=180,
                             fill_mode='reflect',                          
                             preprocessing_function=pre_func)   

# Batch Size
batch_size = 64

# * Layer Configuration
for layer in model.layers:
    layer.trainable = True

# Compile the model
loss_ = losses.binary_crossentropy
optimizer_ = optimizers.SGD(lr=0.01, momentum=0.9)
metrics_ = [f_beta_k_, 'accuracy']
model.compile(loss=loss_, optimizer=optimizer_, metrics=metrics_)

# Fit
model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=batch_size), 
                    validation_data=(X_val, y_val),  
                    epochs=100,
                    steps_per_epoch=len(X_tr)/batch_size,                    
                    callbacks=callback_list)  


# Do prediction of the validation set
p = model.predict(X_val, verbose=1)

# Save predictions
np.save('./model_inception_1/p.npy', p)
print 'Validation predictions are saved.'

# ** Threshold Optimization **
# Find max score and plot (global thresholding)
print 'global t:'
scores = []
for t in np.linspace(0, 1, 100):
    f_score = fbeta_score(y_val, p > t, beta=2, average='samples')
    scores.append((t, f_score))    
scores = sorted(scores, key=lambda x: x[1], reverse=True)
OPTIMUM_GLOBAL_T = scores[0][0]
print 'Max. Score:', scores[0][1]
print 't:', scores[0][0]
os.write(1, 'Max. Score: ' + str(scores[0][1]) + '\n')
os.write(1, 't: ' + str(scores[0][0]) + '\n')


# Find max score (seperate thresholding)
print 'seperate t:'
optimum_thresholds = []
for col in range(p.shape[1]):
    thresholded = p > 0.5
    scores = []
    for t in np.linspace(0, 1, 100):
        thresholded[:, col] = p[:, col] > t
        # Get the score
        score = metrics.fbeta_score(y_val, (thresholded).astype(np.int), beta=2, average='samples')
        # Append the threshold and the result
        scores.append((t, score))
    # Find the best score with its threshold
    scores.sort(key=lambda x: x[1], reverse=True)
    optimum_thresholds.append(scores[0][0])       
# Get the score with optimum thresholds
score = metrics.fbeta_score(y_val, (p > optimum_thresholds).astype(np.int), beta=2, average='samples')
print 'score:', score
_ = os.write(1, 'Test_score:' + str(score) + '\n')

print 'Done.'