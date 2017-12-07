# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 16:37:04 2017

@author: msunasra
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import Imputer



#class DataPreprocess(object):
    

def load_data(path):
    df = pd.read_csv(path,sep=';')
    #Dropping duration feature as it's highly dependent
    df.drop('duration',axis=1,inplace=True)
    return df       

def convertOutputFeatures(df):
    df['y'] = df['y'].map({'no':0,'yes':1})
    return df

def convertInputFeatures(df):
	#Grouping Education variable
	df_basic_ed = df[(df.education == 'basic.4y') | (df.education == 'basic.6y') | (df.education == 'illiterate')]
	df_mid_ed = df[(df.education == 'basic.9y') | (df.education == 'high.school')]
	df_degree_ed = df[(df.education == 'professional.course') | (df.education == 'university.degree')]
	        
	df.loc[df_basic_ed.index, 'education'] = 'Basic'
	df.loc[df_mid_ed.index, 'education'] = 'Mid'
	df.loc[df_degree_ed.index, 'education'] = 'Degree'
	 
	df_unskilled = df[(df.job == 'blue-collar') | (df.job == 'housemaid')]
	df_service = df[(df.job == 'admin.') | (df.job == 'services') | (df.job == 'technician')]
	df_professional = df[(df.job == 'entrepreneur') | (df.job == 'self-employed') | (df.job == 'management')]
	df_student = df[(df.job == 'student')]
	df_retired = df[(df.job == 'retired')]
	df_unemployed = df[(df.job == 'unemployed')]      
	    
	df.loc[df_unskilled.index, 'job'] = 'Unskilled'
	df.loc[df_service.index, 'job'] = 'Service'
	df.loc[df_professional.index, 'job'] = 'Professional'
	df.loc[df_student.index, 'job'] = 'Student'
	df.loc[df_retired.index, 'job'] = 'Retired'
	df.loc[df_unemployed.index, 'job'] = 'Unemployed'


	#Imputing Unknown marital status    
	df.loc[(df['marital']  == 'unknown') & (df['age']  >= 40), 'marital'] = 'married'
	df.loc[(df['marital']  == 'unknown') & (df['age']  < 40), 'marital'] = 'single'

	#Converting contact into binary      
	df['contact'] = df['contact'].map({'telephone':0,'cellular':1})

	#Removing insignificant variables
	df.drop(['day_of_week','pdays','month'],axis=1,inplace=True)

	return df

def scaleEconomicFeatures(df,cols_transform = ['emp.var.rate', 'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']):
    scaler = StandardScaler()
    scaler.fit_transform(df[cols_transform])    
    df.loc[:, cols_transform] = scaler.fit_transform(df.loc[:, cols_transform])
    
    return df

def convertDummies(df):

    df = pd.get_dummies(df)
    return df

def split_data(df,target='y'):
	
    df_X = df.drop(target,axis=1)
    y = df[target]
    return df_X, y
