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

def xy_split(df, target='y'):
    '''
    Creates X matrix and y vector from df dataframe
    return X, y
    '''
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y
    
def tt_split(X, y, test_size=0.5, random_state=42):
    '''
    Performs train test split 
    returns X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def run_model(X_train, X_test, y_train, y_test, model):
    '''
    Performs model training and tests using ROC-AUC 
    returns AUC score
    '''
    model.fit(X_train, y_train)
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    return auc

def build_model(X_train, X_test, y_train, y_test, model):
    '''
    Performs model training and tests using ROC-AUC 
    returns AUC score
    '''
    model = model()
    model.fit(X_train, y_train)
    return model

def con_cat_split(X, con_cols, cat_cols):
    '''
    Performs dataframe splits based on the continous & 
    categorical column lists passed as arguments
    
    returns con_df & cat_df
    '''
    con_df = X[con_cols]
    cat_df = X[cat_cols]
    return con_df, cat_df

def chi2_test(cat_df, X, y, k=10):
    '''
    Performs chi2 feature importance/significance test
    takes categorical variables as input 
    returns chi2_df
    '''
    Xd = pd.get_dummies(X)
    cat_df_oh = pd.get_dummies(cat_df)
    skb = SelectKBest(chi2,k=k)
    skb.fit(cat_df_oh, y)
    chi2_df = Xd[cat_df_oh.columns[skb.get_support()]]
    
    chi2_test_df = pd.DataFrame(skb.scores_,columns=['chi2 score'])
    chi2_test_df['pvals'] = skb.pvalues_
    chi2_test_df.sort_values(by='pvals',ascending=True)

    return chi2_df, chi2_test_df

def oversample(X_train, y_train):
    '''
    Performs oversampling using SMOTE on train data
    returns oversampled X_res, y_res
    '''
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X_train, y_train)
    return X_res, y_res

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

df = pd.read_csv('bank-additional/bank-additional.csv', sep=';', index_col=0, doublequote=True) # Read CSV file to df pandas dataframe 
df['y'] = np.where(df['y']=='yes',1,0) # Encode y target as 0 & 1
df.drop(['duration'],axis=1,inplace=True) # Drop Duration feature as it is biased
df.head() # displays head of dataframe

# X y split
X, y = xy_split(df, target='y')
# One Hot Encode 
Xd = pd.get_dummies(X)
# Standard Scaling 
Xd = scale(Xd)
# Train Test split
X_train, X_test, y_train, y_test = tt_split(Xd, y)

models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

for model in models:
    # run model
    model = model()
    auc = run_model(X_train, X_test, y_train, y_test, model) # train and returns AUC test score
    print('AUC Score = %.2f' %(auc*100) +' %\nOn Model - \n'+str(model))

# Oversample using SMOTE
X_res, y_res = oversample(X_train, y_train)
# Oversampling using smote
yr = pd.DataFrame(y_res,columns=['y'])
yr['y'].value_counts().plot.bar()

# Models as list
models = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

# Run each model
for model in models:
    # run model
    model = model()
    auc = run_model(X_res, X_test, y_res, y_test, model) # train and returns AUC test score
    print('AUC Score = %.2f' %(auc*100) +' %\nOn Model - \n'+str(model))

print(X.columns)
cat_cols = ['job', 'marital', 'education', 'default', 
            'housing', 'loan','contact', 'month', 'day_of_week', 
            'poutcome']

con_cols = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
           'euribor3m', 'nr.employed']
con_df, cat_df = con_cat_split(X, con_cols, cat_cols)
chi2_df, chi2_test_df = chi2_test(cat_df, X, y, k=35)
chi2_test_df.sort_values(by=['pvals'])[:10]

# In[]:

import pickle
filename = 'model_v1.pk'

model = build_model(X_res, X_test, y_res, y_test, RandomForestClassifier)

with open(filename, 'wb') as file:
	pickle.dump(model, file)
