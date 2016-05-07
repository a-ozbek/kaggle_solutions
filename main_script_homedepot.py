import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def get_rms_error(y,y_pred):
    return mean_squared_error(y, y_pred)**0.5


n_train = 74067
n_test = 166693

df = pd.read_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/homedepot/train_test_clean.csv")

df_train = df[:n_train]
df_test = df[n_train:]

df_train_id = df_train['id']
df_test_id = df_test['id']

y_train = df_train['relevance']
X_train = df_train.drop(['id','relevance'], axis = 1)

X_test = df_test.drop(['id','relevance'], axis = 1)

#Train Regression Model

# #SVR
# from sklearn.svm import SVR
# svr_rbf_model = SVR()
# svr_rbf_model =  svr_rbf_model.fit(X_train, y_train)
# y_train_predicted = svr_rbf_model.predict(X_train)

# # Ridge Regression
# from sklearn.linear_model import Ridge
# ridge_model = Ridge(alpha = 2)
# ridge_model.fit(X_train,y_train)
# y_train_predicted = ridge_model.predict(X_train)
# y_test_predicted = ridge_model.predict(X_test)

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10)
rf_model.fit(X_train, y_train)
y_train_predicted = rf_model.predict(X_train)
y_test_predicted = rf_model.predict(X_test)

print "Train Error: ", get_rms_error(y_train,y_train_predicted)

#Write to file
df_test_predicted = pd.DataFrame()
df_test_predicted['id'] = df_test_id
df_test_predicted['relevance'] = y_test_predicted
df_test_predicted.columns = ['id','relevance']

#Write
df_test_predicted.to_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/homedepot/test_predicted.csv",index = False)


