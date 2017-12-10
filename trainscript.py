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
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
#from sklearn.grid_search import
from sklearn.metrics import roc_auc_score 


def oversample(X, y):
    '''
    Performs oversampling using SMOTE on train data
    returns oversampled X_res, y_res
    '''
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X, y)
    return X_res, y_res

def tt_split(X, y, test_size=0.3, random_state=42):
    '''
    Performs train test split 
    returns X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_model_score(model, X_train, X_test, y_train, y_test):
    '''
    Performs model training and returns AUC score
    '''
    model = model.fit(X_train, y_train)
    ypred = model.predict_proba(X_test)
    
    return roc_auc_score(y_test,ypred[:,1])

def model_using_cross_val(model, X, y, cv=5,scoring='roc_auc'):
    '''
    Performs model training using 5 fold cross validation and returns AUC score
    '''
    #model = model()
    scores = cross_val_score(model,X,y, cv=cv,scoring=scoring)
    return scores.mean()

def run_models(X_train, X_test, y_train, y_test):
    
    lr = LogisticRegression(random_state=9)
    dt = DecisionTreeClassifier(random_state=9)
    rfc = RandomForestClassifier(random_state=9)

    models = [lr,dt,rfc]

    #Run each model
    #model_scores = dict()
    scores = []
    for model in models:
        # run model
        #model = model()
        auc = get_model_score(model, X_train, X_test, y_train, y_test) # train and returns AUC test score
        
        #model_scores[str(model)] = auc
        scores.append(auc)
    return scores

def run_rfc(X_train, X_test, y_train, y_test):
    rfc = LogisticRegression(random_state=9)
    proba = get_proba(rfc, X_train, X_test, y_train, y_test)
    return proba


def get_proba(model, X_train, X_test, y_train, y_test):
    '''
    Performs model training and returns AUC score
    '''
    model = model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    ypred_proba = model.predict_proba(X_test)
    return ypred,ypred_proba
