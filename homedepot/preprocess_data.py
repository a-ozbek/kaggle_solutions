import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from aux_funcs import *

# Read datasets
df_train = pd.read_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/homedepot/train.csv")
df_test = pd.read_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/homedepot/test.csv")
df_prod_desc = pd.read_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/homedepot/product_descriptions.csv")

# Merging
n_train = (df_train.shape)[0]
n_test = (df_test.shape)[0]
print "n_train: ", n_train
print "n_test: ", n_test

#Merge train and test
df_train_test = pd.concat([df_train, df_test],axis = 0, ignore_index=True)
#Merge product description to the corresponding item
df_train_test = pd.merge(df_train_test,df_prod_desc, how='left', on = 'product_uid')

# #Take a subsample
# df_train_test = pd.concat([df_train_test[0:20],df_train_test[75000:75030]], axis = 0, ignore_index=True)

#Clean the text in the dataset
df_train_test['product_title'] = df_train_test['product_title'].apply(clean_text)
print "'procuct_title' cleaning done."

df_train_test['search_term'] = df_train_test['search_term'].apply(clean_text)
print "'search_term' cleaning done."

df_train_test['product_description'] = df_train_test['product_description'].apply(clean_text)
print "'product_description' cleaning done."

#Calculate word frequencies
df_train_test['product_merged'] = df_train_test['search_term'] + "," + \
                                  df_train_test['product_title'] + "," + \
                                  df_train_test['product_description']

df_train_test['n_product_title'] = df_train_test['product_merged'].map(lambda x: count_words(search_term=x.split(',')[0] , text=x.split(',')[1]))
df_train_test['n_product_description'] = df_train_test['product_merged'].map(lambda x: count_words(search_term=x.split(',')[0] , text=x.split(',')[2]))


#Drop unnecessary columns
df_train_test = df_train_test.drop(['product_title','product_uid','search_term','product_description','product_merged'],axis = 1)

#Write to file
df_train_test.to_csv("/Users/ahmetcanozbek/Desktop/kaggleDatasets/homedepot/train_test_clean.csv", index = False)








