import numpy as np
import pandas as pd

def get_error_rate(y_true,y_predicted):
    return float(sum(y_true != y_predicted)) / (y_true.size)


df = pd.read_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/titanic/train.csv", header = 0)
df_test = pd.read_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/titanic/test.csv", header = 0)

# Drop: Name, Ticket, Cabin
df = df.drop(['Name','Ticket','Cabin','PassengerId'], axis = 1)

df_test_passengerid = df_test['PassengerId']
df_test = df_test.drop(['Name','Ticket','Cabin','PassengerId'], axis = 1)

#Turn 'Sex','Embarked' feature into one hot coding
df = pd.get_dummies(df,columns=['Sex','Embarked','Pclass'])
df_test = pd.get_dummies(df_test, columns=['Sex','Embarked','Pclass'])

#Handle NaN values, fill with median method
df = df.fillna(df.median())
df_test = df_test.fillna(df_test.median())

#split label
y_train = df['Survived']
x_train = df.drop(['Survived'], axis = 1)

x_test = df_test

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Train
random_forest_model = RandomForestClassifier(n_estimators = 10)
random_forest_model = random_forest_model.fit(x_train,y_train)
y_train_predicted = random_forest_model.predict(x_train)
print "Train Error Rate: ", get_error_rate(y_train,y_train_predicted)

#Test
y_test_predicted = random_forest_model.predict(x_test)
#Convert numpy array of test prediction to pandas data frame
y_test_predicted = pd.DataFrame(y_test_predicted)


#Write test predictions to csv file
df_test_predicted = pd.concat([df_test_passengerid,y_test_predicted],axis = 1)
df_test_predicted.columns = ['PassengerId','Survived']
#Write
df_test_predicted.to_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/titanic/test_predicted.csv", index = False)













