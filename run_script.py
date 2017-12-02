# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:56:37 2017

@author: sidpa
"""
import pandas as pd
import pickle
filename = 'model_v1.pk'
with open(filename ,'rb') as f:
    loaded_model = pickle.load(f)

test_df = pd.read_csv('test_df.csv')

def get_prediction(loaded_model, test_df):
    return loaded_model.predict_proba(test_df)

print(get_prediction(loaded_model, test_df))
