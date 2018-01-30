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
    im_bandavg = ((im_band1 * im_band2) / 2) ** 0.5
    im_bandavg[np.isnan(im_bandavg)] = 0

    # - Zero mean
    im_band1 -= np.mean(im_band1)
    im_band2 -= np.mean(im_band2)
    im_bandavg -= np.mean(im_bandavg)

    # - Normalize
    im_band1 /= np.std(im_band1)
    im_band2 /= np.std(im_band2)
    im_bandavg /= np.std(im_bandavg)

    im_band1 = np.array(im_band1).reshape(75, 75, 1)
    im_band2 = np.array(im_band2).reshape(75, 75, 1)
    im_bandavg = np.array(im_bandavg).reshape(75, 75, 1)
    im_star = (np.expand_dims(transform.rotate(im_bandavg, 90), axis=0) + np.expand_dims(
        transform.rotate(im_bandavg, 180), axis=0) + \
               np.expand_dims(transform.rotate(im_bandavg, 270), axis=0) + im_bandavg)**2
    im_star = np.array(im_star).reshape(75, 75, 1)

    #Normalize
    im_star -= np.mean(im_star)
    im_star /= np.std(im_star)

    # im_band1[::-1,:,:] #+ im_bandavg
    # im = np.concatenate([im_band1, im_band2,im_bandavg,im_hflip], axis=2)

    im = np.concatenate([im_band1, im_band2, im_star], axis=2)

    X.append(im)
    y.append(label)

#X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], ((x_band1+x_band1)/2)[:, :, :, np.newaxis]], axis=-1)
X = np.array(X)
y = np.array(y)
print ('X.shape: '+ str(X.shape))
print ('y.shape: '+ str(y.shape))

#X_TEST
df_test = pd.read_json('test.json')
X_test, y_test = [], []
for im_band1, im_band2 in zip(df_test['band_1'], df_test['band_2']):
    im_band1 = np.array(im_band1).reshape(75, 75, 1)
    im_band2 = np.array(im_band2).reshape(75, 75, 1)
    im_bandavg = ((im_band1 * im_band2) / 2) ** 0.5
    im_bandavg[np.isnan(im_bandavg)] = 0

    # - Zero mean
    im_band1 -= np.mean(im_band1)
    im_band2 -= np.mean(im_band2)
    im_bandavg -= np.mean(im_bandavg)

    # - Normalize
    im_band1 /= np.std(im_band1)
    im_band2 /= np.std(im_band2)
    im_bandavg /= np.std(im_bandavg)

    im_band1 = np.array(im_band1).reshape(75, 75, 1)
    im_band2 = np.array(im_band2).reshape(75, 75, 1)
    im_bandavg = np.array(im_bandavg).reshape(75, 75, 1)
    im_star = (np.expand_dims(transform.rotate(im_bandavg, 90), axis=0) + np.expand_dims(
        transform.rotate(im_bandavg, 180), axis=0) + \
                 np.expand_dims(transform.rotate(im_bandavg, 270), axis=0) + im_bandavg) ** 2
    im_star = np.array(im_star).reshape(75, 75, 1)

    # Normalize
    im_star -= np.mean(im_star)
    im_star /= np.std(im_star)

     # im_band1[::-1,:,:] #+ im_bandavg
    # im = np.concatenate([im_band1, im_band2,im_bandavg,im_hflip], axis=2)

    im = np.concatenate([im_band1, im_band2, im_star], axis=2)

    X_test.append(im)
X_test = np.array(X_test)
print('X_test.shape:' + str(X_test.shape))

def arrange_datas(MODEL_NUMBER):
    MODEL_PATH = '.model_star' + str(MODEL_NUMBER) + '.h5'
    model = load_model(MODEL_PATH)
    print("Train is being prepared..")
    model_new = Model(inputs=model.input, outputs=model.layers[-4].output)
    train_xgb= model_new.predict((X), verbose=1)
    train_xgb=pd.DataFrame(train_xgb)
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    train_xgb['angle'] =df['inc_angle']
    train_xgb['angle'] =train_xgb['angle'].fillna(train_xgb['angle'].median())
    train_xgb=np.array(train_xgb)

    print("\nTest is being prepared..")
    test_xgb= model_new.predict((X_test), verbose=1)
    test_xgb=np.array(test_xgb)
    test_xgb=pd.DataFrame(test_xgb)
    df_test['inc_angle'] = pd.to_numeric(df_test['inc_angle'], errors='coerce')
    test_xgb['angle'] =df_test['inc_angle']
    test_xgb=np.array(test_xgb)
    return train_xgb,test_xgb

# Train - Val SPlit
N_SPLITS = 5
mn_list = [1,2,3,4,5]
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
#from sklearn import ensemble, metrics, model_selection, naive_bayes

xgb_results= []
logreg_results= []
xgb_result_all = []
logreg_results_all= []
pred_test_xgb=0
pred_test_lr = 0
for MODEL_NUMBER in mn_list:
    print('MODEL_NUMBER:'+str(MODEL_NUMBER))
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_SEED, shuffle=True)
    train_xgb,test_xgb = arrange_datas(MODEL_NUMBER)
    cv = list(skf.split(train_xgb, y))
    train_i, val_i = cv[MODEL_NUMBER - 1]
    X_train, y_train = train_xgb[train_i], y[train_i]
    X_val, y_val = train_xgb[val_i], y[val_i]

    print ('X_train.shape:'+ str(X_train.shape))
    print ('y_train.shape:'+ str(y_train.shape))
    print ('X_val.shape:'+ str(X_val.shape))
    print ('y_val.shape:' +  str(y_val.shape))
    print ('np.mean(y_train):'+ str(np.mean(y_train)))
    print ('np.mean(y_val):'+ str(np.mean(y_val)))

    #from sklearn import ensemble, metrics, model_selection, naive_bayes

    #model_nb = naive_bayes.MultinomialNB()
    #model_nb.fit(X_train, y_train)
    #X_train = pd.DataFrame(X_train)
    #X_train['nb'] = model_nb.predict(X_train)
    #X_train = np.array(X_train)

    #X_val = pd.DataFrame(X_val)
    #X_val['nb'] = model_nb.predict(X_val)
    #X_val = np.array(X_val)

    dx1 = xgb.DMatrix(X_train, y_train)
    dx2 = xgb.DMatrix(X_val, y_val)

    watchlist = [(dx2, 'valid')]
    print("XGB modeling has started..")
    # depth=4 best

    params = {'eta': 0.01, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.01, 'objective': 'binary:logistic',
              'eval_metric': 'logloss', 'seed': 99, 'silent': True,'reg_lambda':0}

    model = xgb.train(params, dx1, 5000, watchlist, maximize=False, verbose_eval=50,
                      early_stopping_rounds=200)

    pred_val_xgb = model.predict(xgb.DMatrix(X_val), ntree_limit=model.best_ntree_limit)
    print('XGB Result:'+str(log_loss(y_val, pred_val_xgb))) # ('gini', 0.28484043572763312)
    xgb_results.append(log_loss(y_val, pred_val_xgb))
    #All
    pred_all_xgb = model.predict(xgb.DMatrix(train_xgb), ntree_limit=model.best_ntree_limit)
    print('XGB Result All:' + str(log_loss(y, pred_all_xgb)))  # ('gini', 0.28484043572763312)
    xgb_result_all.append(log_loss(y, pred_all_xgb))
    #Test
    pred_test_xgb += model.predict(xgb.DMatrix(test_xgb), ntree_limit=model.best_ntree_limit) / len(mn_list)


    print('LogisticRegression has started..')
    lr = LogisticRegression(class_weight='balanced',penalty='l2', C=0.014)

    lr.fit(X_train, y_train)
    pred_val_lr = lr.predict_proba(X_val)[:, 1]
    print('LogReg Result: '+str(log_loss(y_val,pred_val_lr)))
    logreg_results.append(log_loss(y_val,pred_val_lr))
    #All
    pred_all_lr = lr.predict_proba(train_xgb)[:, 1]
    print('LogReg Result All: '+str(log_loss(y,pred_all_lr)))
    logreg_results_all.append(log_loss(y,pred_all_lr))
    #Test
    pred_test_lr += lr.predict_proba(test_xgb)[:, 1] / len(mn_list)

print(xgb_results)
print('Mean:'+str(np.mean(xgb_results))+ ' Std:'+str(np.std(xgb_results)))

print(xgb_result_all)
print('Mean:'+str(np.mean(xgb_result_all))+ ' Std:'+str(np.std(xgb_result_all)))


print(logreg_results)
print('Mean:'+str(np.mean(logreg_results[0],logreg_results[2],logreg_results[4],logreg_results[6],logreg_results[8]))+
      ' Std:'+str(np.std(logreg_results[0]+logreg_results[2]+logreg_results[4]+logreg_results[6]+logreg_results[8])))

print(xgb_result_all)
print('Mean:'+str(np.mean(logreg_results[5:]))+ ' Std:'+str(np.std(logreg_results[5:])))


df_sub_xgb = pd.DataFrame()
df_sub_xgb['id'] = df_test['id']
df_sub_xgb['is_iceberg'] = pred_test_xgb


df_sub_lr = pd.DataFrame()
df_sub_lr['id'] = df_test['id']
df_sub_lr['is_iceberg'] = pred_test_lr

df_sub_xgb.to_csv('./substar_xgboost_test'+ '.csv', index=False)
df_sub_lr.to_csv('./substar_logreg_test'+ '.csv', index=False)







