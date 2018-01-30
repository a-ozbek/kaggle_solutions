import numpy as np
import pandas as pd
from skimage import transform
from keras.models import Sequential, load_model
from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras import losses, optimizers, callbacks
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold, KFold
from scipy import fftpack
from matplotlib import pyplot as plt

RANDOM_SEED = 43
RANDOM_SEED_XGBOOST = 43 #1453 Fatih's seed
np.random.seed(RANDOM_SEED)

df = pd.read_json('train.json')
df.head()


#ax = fig.add_subplot(3,3,1)
#ax.imshow(im_band1[:, :, 0])
#ax = fig.add_subplot(3,3,2)
#ax.imshow(im_band2[:, :, 0])
#ax = fig.add_subplot(3,3,3)
#ax.imshow(im_bandavg[:, :, 0])
#ax = fig.add_subplot(3,3,4)
#ax.imshow(im_hflip[:, :, 0])

X, y = [], []
for im_band1, im_band2, label in zip(df['band_1'], df['band_2'], df['is_iceberg']):
    im_band1 = np.array(im_band1).reshape(75, 75, 1)
    im_band2 = np.array(im_band2).reshape(75, 75, 1)
    # Preprocess
    # - Zero mean
    im_band1 -= np.mean(im_band1)
    im_band2 -= np.mean(im_band2)
    # - Normalize
    im_band1 /= np.std(im_band1)
    im_band2 /= np.std(im_band2)
    im = np.concatenate([im_band1, im_band2], axis=2)
    X.append(im)
    y.append(label)

X = np.array(X)
y = np.array(y)
print ('X.shape: '+ str(X.shape))
print ('y.shape: '+ str(y.shape))

#Data Augmentation
def bypass(x):
    return x

def h_flip(x):
    return x[:, :, ::-1, :]

def v_flip(x):
    return x[:, ::-1, :, :]

def hv_flip(x):
    return h_flip(v_flip(x))

def rot90(x):
    return np.concatenate([np.expand_dims(transform.rotate(im, 90), axis=0) for im in x], axis=0)

def rot180(x):
    return np.concatenate([np.expand_dims(transform.rotate(im, 180), axis=0) for im in x], axis=0)

def rot270(x):
    return np.concatenate([np.expand_dims(transform.rotate(im, 270), axis=0) for im in x], axis=0)

def rot45(x):
    return np.concatenate([np.expand_dims(transform.rotate(im, 45, mode='reflect'), axis=0) for im in x], axis=0)

def rot135(x):
    return np.concatenate([np.expand_dims(transform.rotate(im, 135, mode='reflect'), axis=0) for im in x], axis=0)

def rot315(x):
    return np.concatenate([np.expand_dims(transform.rotate(im, 315, mode='reflect'), axis=0) for im in x], axis=0)

aug_funcs = [bypass,
             h_flip, v_flip, hv_flip,
             rot90, rot180, rot270]


# Train



#
print ('X_train.shape:'+ str(X.shape))
print ('y_train.shape:'+ str(y.shape))

#X_TEST
df_test = pd.read_json('test.json')
X_test, y_test = [], []
for im_band1, im_band2 in zip(df_test['band_1'], df_test['band_2']):
    im_band1 = np.array(im_band1).reshape(75, 75, 1)
    im_band2 = np.array(im_band2).reshape(75, 75, 1)
    # Preprocess
    # - Zero mean
    im_band1 -= np.mean(im_band1)
    im_band2 -= np.mean(im_band2)
    # - Normalize
    im_band1 /= np.std(im_band1)
    im_band2 /= np.std(im_band2)
    im = np.concatenate([im_band1, im_band2], axis=2)

    X_test.append(im)
X_test = np.array(X_test)
print('X_test.shape:' + str(X_test.shape))

def arrange_datas(MODEL_NUMBER,X_train,X_val,y_train,y_val):
    MODEL_PATH = './ahmet_models/model' + str(MODEL_NUMBER) + '.h5'
    #import h5py # to fix loading model problem
    #f = h5py.File(MODEL_PATH, 'r+')
    #del f['optimizer_weights']
    #f.close()
    X_train = np.concatenate([func(X_train) for func in aug_funcs], axis=0)
    y_train = np.concatenate([y_train] * len(aug_funcs))

    # Validation
    #X_val = np.concatenate([func(X_val) for func in aug_funcs], axis=0)
    #y_val = np.concatenate([y_val] * len(aug_funcs))
    model = load_model(MODEL_PATH)

    print("Train is being prepared..")
    model_new = Model(inputs=model.input, outputs=model.layers[-5].output)
    X_train= model_new.predict((X_train),batch_size=240, verbose=0)
    #model.layers
    print('\n New train shape: '+str(X_train.shape))
    X_train=pd.DataFrame(X_train)
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    X_train['angle'] =df.loc[:len(y_train),'inc_angle']
    X_train['angle'] =X_train['angle'].fillna(df['inc_angle'].median())
    X_train=np.array(X_train)

    print("Val is being prepared..")
    X_val = model_new.predict((X_val), batch_size=240, verbose=0)
    # model.layers
    print('\n New val shape: ' + str(X_val.shape))
    X_val = pd.DataFrame(X_val)
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    X_val['angle'] = df.loc[len(y_train):,'inc_angle']
    X_val['angle'] = X_val['angle'].fillna(df['inc_angle'].median())
    X_val = np.array(X_val)


    print("\nTest is being prepared..")
    test_xgb= model_new.predict((X_test), batch_size=240,verbose=0)
    print('\n New test shape: '+str(test_xgb.shape))
    test_xgb=np.array(test_xgb)
    test_xgb=pd.DataFrame(test_xgb)
    df_test['inc_angle'] = pd.to_numeric(df_test['inc_angle'], errors='coerce')
    test_xgb['angle'] =df_test['inc_angle']
    test_xgb=np.array(test_xgb)
    return X_train,X_val,test_xgb,y_train,y_val

# Train - Val SPlit
N_SPLITS = 5
mn_list = [1,2,3,4,5]
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
#from sklearn import ensemble, metrics, model_selection, naive_bayes

xgb_results= []
MODEL_NUMBER_xgb_results = {}
MODEL_NUMBER_lr_results = {}

logreg_results=[]
xgb_result_all = []
logreg_results_all= []
pred_test_xgb=0
pred_test_lr = 0
for MODEL_NUMBER in mn_list:
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED_XGBOOST, shuffle=True)

    for fold in range(N_SPLITS):
        print('\nMODEL_NUMBER:'+str(MODEL_NUMBER) +' Fold: '+str(fold))
        cv = list(skf.split(X, y))
        train_i, val_i = cv[fold]
        X_train, y_train = X[train_i], y[train_i]
        X_val, y_val = X[val_i], y[val_i]
        X_train, X_val, test_xgb,y_train,y_val = arrange_datas(MODEL_NUMBER,X_train,X_val,y_train,y_val)

        print('X_train.shape:' + str(X_train.shape))
        print('y_train.shape:' + str(y_train.shape))
        print('X_val.shape:' + str(X_val.shape))
        print('y_val.shape:' + str(y_val.shape))
        print('np.mean(y_train):' + str(np.mean(y_train)))
        print('np.mean(y_val):' + str(np.mean(y_val)))

        # from sklearn import ensemble, metrics, model_selection, naive_bayes

        # model_nb = naive_bayes.MultinomialNB()
        # model_nb.fit(X_train, y_train)
        # X_train = pd.DataFrame(X_train)
        # X_train['nb'] = model_nb.predict(X_train)
        # X_train = np.array(X_train)

        # X_val = pd.DataFrame(X_val)
        # X_val['nb'] = model_nb.predict(X_val)
        # X_val = np.array(X_val)

        dx1 = xgb.DMatrix(X_train, y_train)
        dx2 = xgb.DMatrix(X_val, y_val)

        watchlist = [(dx2, 'valid')]
        print("XGB modeling has started..")
        # depth=4 best

        params = {'eta': 0.01, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.01,
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss', 'seed': 99, 'silent': True, 'reg_lambda': 0}

        model = xgb.train(params, dx1, 5000, watchlist, maximize=False, verbose_eval=100,
                          early_stopping_rounds=200)

        pred_val_xgb = model.predict(xgb.DMatrix(X_val), ntree_limit=model.best_ntree_limit)
        print('XGB Result:' + str(log_loss(y_val, pred_val_xgb)))  # ('gini', 0.28484043572763312)
        xgb_results.append(log_loss(y_val, pred_val_xgb))
        # All


        # Test
        pred_test_xgb += model.predict(xgb.DMatrix(test_xgb), ntree_limit=model.best_ntree_limit) / (N_SPLITS*len(mn_list))

        print('LogisticRegression has started..')
        lr = LogisticRegression(class_weight='balanced', penalty='l2', C=0.014)

        lr.fit(X_train, y_train)
        pred_val_lr = lr.predict_proba(X_val)[:, 1]
        print('LogReg Result: ' + str(log_loss(y_val, pred_val_lr)))
        logreg_results.append(log_loss(y_val, pred_val_lr))
        # All

        # Test
        pred_test_lr += lr.predict_proba(test_xgb)[:, 1] / (N_SPLITS*len(mn_list))

    MODEL_NUMBER_xgb_results[MODEL_NUMBER] = xgb_results
    MODEL_NUMBER_lr_results[MODEL_NUMBER] = logreg_results
    xgb_results=[]
    logreg_results=[]

for m in mn_list:
    print("\nMODEL_NUMBER:"+str(m))
    #print("XGB Results: ")
    #print('Mean: '+str(np.mean(MODEL_NUMBER_xgb_results[m])))
    #print('Std: '+str(np.std(MODEL_NUMBER_xgb_results[m])))

    print("\nLogReg Results: ")
    print('Mean: ' + str(np.mean(MODEL_NUMBER_lr_results[m])))
    print('Std: ' + str(np.std(MODEL_NUMBER_lr_results[m])))




df_sub_xgb = pd.DataFrame()
df_sub_xgb['id'] = df_test['id']
df_sub_xgb['is_iceberg'] = pred_test_xgb


df_sub_lr = pd.DataFrame()
df_sub_lr['id'] = df_test['id']
df_sub_lr['is_iceberg'] = pred_test_lr

df_sub_xgb.to_csv('./ahmetsmodels_seperate_agg_xgboost5FOLD_test'+ '.csv', index=False)
df_sub_lr.to_csv('./ahmetsmodels_seperate_agg_logreg5FOLD_test'+ '.csv', index=False)








