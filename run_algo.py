import numpy as np, pandas as pd


#from trainscript.ModelBuilding import *
from data_processing import *
from trainscript import *


#from DataPreprocess import *
path = 'bank-additional/bank-additional-full.csv'

#d = DataPreprocess()
    #d = DataPreprocess();
df1 = load_data(path)
df_converted = convertInputFeatures(df1.copy())
df = convertOutputFeatures(df_converted)
df = convertDummies(df)
df = scaleEconomicFeatures(df)

X, y = split_data(df)


X_train, X_test, y_train, y_test = tt_split(X, y)

X_train_res, y_train_res = oversample(X_train, y_train)

#scores = run_models_with_kfold(X_train_res, y_train_res)

scores = run_models(X_train_res, X_test, y_train_res, y_test)
# print scores
label, proba = run_rfc(X_train_res, X_test, y_train_res, y_test)

df_converted['duration'] = df1['duration']
#proba = run_rfc2(X_train_res, X_test, y_train_res, y_test)
dump_df = df_converted.loc[X_test.index].copy()
dump_df['y_pred'] = label
dump_df['y_proba'] = [n for m,n in proba]
#dump_df['duration'] = df1['duration']
dump_df.to_csv('result_new.csv')
