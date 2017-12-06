import pandas as pd, numpy as np # for data manipulation and linear algebra
import matplotlib.pyplot as plt, seaborn as sns # data visualization libs
from sklearn.model_selection import train_test_split # performs train test split
from sklearn.metrics import roc_auc_score # AUC score metric
from sklearn.feature_selection import SelectKBest, chi2 # chi2 test 
from sklearn.ensemble import RandomForestClassifier # RFC for model training and feature importance
from sklearn.tree import DecisionTreeClassifier # Decision Tree classifier
from sklearn.linear_model import LogisticRegression # Linear Model for classification
from sklearn.preprocessing import StandardScaler # Standard scaler for continous variables
from imblearn.over_sampling import SMOTE # Tool for Oversampling 
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV


#load the data in file into a DataFrame
df = pd.read_csv('bank-additional/bank-additional.csv', sep=';', index_col=0, doublequote=True) # Read CSV file to df pandas dataframe 

df['y'] = np.where(df['y']=='yes',1,0) # Encode y target as 0 & 1

df.drop(['duration'],axis=1,inplace=True) # Drop Duration feature as it is biased
df.head() # displays head of dataframe


def xy_split(df, target='y'):
    '''
    Creates X features matrix and y target vector from a dataframe
    returns X, y
    '''
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y

def scale(Xd, cols_transform = ['emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed']):
    '''
    Performs Standard Scaler given dataframe with continous variables as list
    returns Xd - (Scaled Dataframe)
    '''
    ss = StandardScaler()
    ss.fit_transform(Xd[cols_transform])    
    Xd.loc[:, cols_transform] = ss.fit_transform(Xd.loc[:, cols_transform])
    return Xd

def oversample(X_train, y_train):
    '''
    Performs oversampling using SMOTE on train data
    returns oversampled X_res, y_res
    '''
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X_train, y_train)
    return X_res, y_res

# def tt_split(X, y, test_size=0.3, random_state=42):
#     '''
#     Performs train test split 
#     returns X_train, X_test, y_train, y_test
#     '''
#     X_train, X_test, y_train, y_test = train_test_split(
#          X, y, test_size=test_size, random_state=random_state)
#     return X_train, X_test, y_train, y_test

def build(model, X, y):
    '''
    Performs model training and tests using ROC-AUC 
    returns AUC score
    '''
    model = model.fit(X, y)
    return model

def run_model(model, X, y, cv=5, scoring='auc'):
    '''
    Performs model training using 5 fold cross validation and returns AUC score
    '''
    model = model()
    score = cross_val_score(model,X, y, cv=cv,scoring=scoring)
    
    return score

# X y split
X, y = xy_split(df, target='y')

# One Hot Encode 
X_dummies = pd.get_dummies(X)

# Standard Scaling 
X_dummies = scale(X_dummies)

# Oversample using SMOTE
X_res, y_res = oversample(X_dummies, y)

# Models as list
models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

# Run each model
model_scores = dict()
for model in models:
    # run model
    model = model()
    auc = run_model(model, X_res, y_res,cv=5,scoring='auc') # train and returns AUC test score
    model_scores[str(model)] = auc
    print('AUC Score = %.2f' %(auc*100) +' %\nOn Model - \n'+str(model))

#Based on which model gives best AUC, use the model to train on the entire dataset and save the model
import pickle
filename = 'model_v1.pk'

model = build_model(RandomForestClassifier, X_res,y_res)

with open(filename, 'wb') as file:
	pickle.dump(model, file)
