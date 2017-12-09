import numpy as np, pandas as pd


#from trainscript.ModelBuilding import *
from data_processing import *
from trainscript import *


#from DataPreprocess import *
path = 'bank-additional/bank-additional-full.csv'

#d = DataPreprocess()
    #d = DataPreprocess();
df = load_data(path)
df = convertInputFeatures(df)
df = convertOutputFeatures(df)
df = convertDummies(df)
df = scaleEconomicFeatures(df)

X, y = split_data(df)


X_train, X_test, y_train, y_test = tt_split(X, y)

X_train_res, y_train_res = oversample(X_train, y_train)

#scores = run_models_with_kfold(X_train_res, y_train_res)

# scores = run_models(X_train_res, X_test, y_train_res, y_test)
# print scores
proba = run_rfc(X_train_res, X_test, y_train_res, y_test)
X_test['y_proba'] = [n for m,n in proba]
print X_test.to_csv('result.csv')