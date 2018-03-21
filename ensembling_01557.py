# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:58:24 2017
Subject : Pover-T Tests: Predicting Poverty
@author: Lucas Dienis
"""

# In[]:
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
import os 
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
import sklearn

from random import randint
#LightGBM
import lightgbm as lgb
#
from tqdm import tqdm
#
from collections import Counter
# MLP with manual validation set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint

# For adding new activation function
from keras import backend as K
from keras import optimizers
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint


from sklearn.preprocessing import StandardScaler

#for balanced datasets
from imblearn.over_sampling import SMOTE

#Plots
from matplotlib import pyplot
from xgboost import plot_importance
#%matplotlib inline
from matplotlib.pylab import rcParams

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Can not import plt')

##############################################################################################################
#                                               PARAMETERS                                                   #
##############################################################################################################
#Plot parameters

#parameter to predict 
target = 'poor'
IDcol = 'id'
IDinv = 'iid'

# In[]
##############################################################################################################
#                                               PATHS                                                        #
##############################################################################################################
path = 'C:/Users/20007488/Documents/Competitions/Poverty/'

data_paths = {'A': {'House':{'train': os.path.join(path, 'A', 'A_hhold_train.csv'), 
                             'test':  os.path.join(path, 'A', 'A_hhold_test.csv')},
                    'Indiv':{'train': os.path.join(path, 'A', 'A_indiv_train.csv'), 
                             'test':  os.path.join(path, 'A', 'A_indiv_test.csv')}}, 
              
              'B': {'House':{'train': os.path.join(path, 'B', 'B_hhold_train.csv'), 
                             'test':  os.path.join(path, 'B', 'B_hhold_test.csv')},
                    'Indiv':{'train': os.path.join(path, 'B', 'B_indiv_train.csv'), 
                             'test':  os.path.join(path, 'B', 'B_indiv_test.csv')}}, 
              
              'C': {'House':{'train': os.path.join(path, 'C', 'C_hhold_train.csv'), 
                             'test':  os.path.join(path, 'C', 'C_hhold_test.csv')},
                    'Indiv':{'train': os.path.join(path,'C','C_indiv_train.csv'),
                             'test':  os.path.join(path,'C','C_indiv_test.csv')}}}

# In[]:
##############################################################################################################
#                                               RETRIEVING HOUSE DATASETS                                    #
##############################################################################################################
#train datasets
df_A_house_train = pd.read_csv(data_paths['A']['House']['train'])
df_B_house_train = pd.read_csv(data_paths['B']['House']['train'])
df_C_house_train = pd.read_csv(data_paths['C']['House']['train'])

#test datasets
df_A_house_test = pd.read_csv(data_paths['A']['House']['test'])
df_B_house_test = pd.read_csv(data_paths['B']['House']['test'])
df_C_house_test = pd.read_csv(data_paths['C']['House']['test'])

# In[]:
##############################################################################################################
#                                               RETRIEVING INDIV DATASETS                                    #
##############################################################################################################
#train datasets
df_A_indiv_train = pd.read_csv(data_paths['A']['Indiv']['train'])
df_B_indiv_train = pd.read_csv(data_paths['B']['Indiv']['train'])
df_C_indiv_train = pd.read_csv(data_paths['C']['Indiv']['train'])

#test datasets
df_A_indiv_test = pd.read_csv(data_paths['A']['Indiv']['test'])
df_B_indiv_test = pd.read_csv(data_paths['B']['Indiv']['test'])
df_C_indiv_test = pd.read_csv(data_paths['C']['Indiv']['test'])


# In[]
def ModifyData(df):
    for x in tqdm(df.columns,total = len(df.columns)):
        if df[x].dtype == 'O':
            a = pd.Series(df[x]).astype('category')
            b = a.cat.codes
            df[x] = b
    return df

# In[]
    #######################  MODIFYING DATA 
#Train
df_A_house_train = ModifyData(df_A_house_train)
df_B_house_train = ModifyData(df_B_house_train)
df_C_house_train = ModifyData(df_C_house_train)

#Test
df_A_house_test = ModifyData(df_A_house_test)
df_B_house_test = ModifyData(df_B_house_test)
df_C_house_test = ModifyData(df_C_house_test)

# In[]:
#replacing NaN in B dataset
df_B_house_train = df_B_house_train.replace(np.nan,0)
df_B_house_test = df_B_house_test.replace(np.nan,0)
# In[]:

##############################################################################################################
#                                               DEFINING PREDICTORS                                          #
##############################################################################################################
#Choose all predictors except target & IDcols
predictors_A_house = [x for x in df_A_house_train.columns if x not in [target, IDcol,'country']]
predictors_B_house = [x for x in df_B_house_train.columns if x not in [target, IDcol,'country']]
predictors_C_house = [x for x in df_C_house_train.columns if x not in [target, IDcol,'country']]

# In[]:
def equilibrate(df,predictors,target):
    sm = SMOTE(random_state = 4242)
    x_resampled, y_resampled = sm.fit_sample(df[predictors],df[target])
    ############
    part_a = pd.DataFrame(x_resampled, columns = predictors)
    part_b = pd.DataFrame(y_resampled, columns = [target])
    frames = [part_a,part_b]
    df_resampled = pd.concat(frames,axis = 1)
    
    return(df_resampled)

# In[]:
df_A_resampled = equilibrate(df_A_house_train,predictors_A_house,target)
df_B_resampled = equilibrate(df_B_house_train,predictors_B_house,target)
df_C_resampled = equilibrate(df_C_house_train,predictors_C_house,target)

# In[]:
def equilibrate_test(df,predictors,target):
    x_resampled, x_not_resampled, y_resampled, y_not_resampled = train_test_split(df[predictors], df[target], test_size=0.1, random_state=4242)
    ############
    sm = SMOTE(random_state = 4242)
    x_resampled, y_resampled = sm.fit_sample(x_resampled,y_resampled)
    ############
    part_a = pd.DataFrame(x_resampled, columns = predictors)
    part_b = pd.DataFrame(y_resampled, columns = [target])
    frames = [part_a,part_b]
    df_resampled = pd.concat(frames,axis = 1)
    ############
    part_a = pd.DataFrame(x_not_resampled, columns = predictors)
    part_b = pd.DataFrame(y_not_resampled, columns = [target])
    frames = [part_a,part_b]
    df_not_resampled = pd.concat(frames,axis = 1)
    return(df_resampled,df_not_resampled)
    
# In[]:
df_A_resampled_test, df_A_not_resampled = equilibrate_test(df_A_house_train,predictors_A_house,target)
df_B_resampled_test, df_B_not_resampled  = equilibrate_test(df_B_house_train,predictors_B_house,target)
df_C_resampled_test, df_C_not_resampled  = equilibrate_test(df_C_house_train,predictors_C_house,target)

# In[]
##############################################################################################################
#                                               XGBOOST                                                      #
##############################################################################################################
params_house_A = {}
params_house_A['objective'] = 'binary:logistic'
params_house_A['eval_metric'] = 'logloss'
params_house_A['eta'] = 0.1
params_house_A['gamma'] = 0.1
params_house_A['max_depth'] = 4
params_house_A['min_child_weight'] = 12
params_house_A['subsample'] = 0.8
params_house_A['colsample_bytree'] = 0.8
params_house_A['alpha'] = 0.001

# In[]
params_house_B = {}
params_house_B['objective'] = 'binary:logistic'
params_house_B['eval_metric'] = 'logloss'
params_house_B['eta'] = 0.3
params_house_B['gamma'] = 0.1
params_house_B['max_depth'] = 1
params_house_B['min_child_weight'] = 7
params_house_B['subsample'] = 0.7
params_house_B['colsample_bytree'] = 0.7
params_house_B['alpha'] = 0.0001
    
# In[]
params_house_C = {}
params_house_C['objective'] = 'binary:logistic'
params_house_C['eval_metric'] = 'logloss'
params_house_C['eta'] = 0.1
params_house_B['gamma'] = 0.2
params_house_C['max_depth'] = 4
params_house_C['min_child_weight'] = 12
params_house_C['subsample'] = 0.9
params_house_C['colsample_bytree'] = 0.9
params_house_C['alpha'] = 1

# In[]
def InitializeIndex(df):
    df.index = np.arange(len(df.index))
    return(df)
    
def XGB_NRS(df_train,df_test,predictors,target,params):
    x_train, x_valid, y_train, y_valid = train_test_split(df_train[predictors], df_train[target], test_size=0.1, random_state=4242)
      
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    model = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    
    dtest = xgb.DMatrix(df_test[predictors])
    
    predictions = model.predict(dtest)
    
    importance = pd.DataFrame(list(model.get_score().items()), columns=['columns', 'values'])# feature importance
    importance = importance.sort_values(['values'], ascending = False)
    importance = InitializeIndex(importance)
    # plot
    plot_importance(model)
    
    return(predictions, importance)
    
def XGB_RS(df_train_resampled,df_train_not_resampled,df_test,predictors,target,params):
      
    d_train = xgb.DMatrix(df_train_resampled[predictors], label=df_train_resampled[target])
    d_valid = xgb.DMatrix(df_train_not_resampled[predictors], label=df_train_not_resampled[target])
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    model = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    
    dtest = xgb.DMatrix(df_test[predictors])

    predictions = model.predict(dtest)
    
    importance = pd.DataFrame(list(model.get_score().items()), columns=['columns', 'values'])# feature importance
    importance = importance.sort_values(['values'], ascending = False)
    importance = InitializeIndex(importance)
    # plot
    plot_importance(model)
    
    return(predictions, importance)

def XGB(df_train,df_test,predictors,target,params,rounds):
    d_train = xgb.DMatrix(df_train[predictors], label=df_train[target])
    
    bst = xgb.train(params, d_train, rounds)
    
    dtest = xgb.DMatrix(df_test[predictors])

    predictions = bst.predict(dtest)
    
    return(predictions)

# In[] XGB TEST NOT RESAMPLED DATASET
Pred_A_H_XGB_NRS, Imp_A_H_XGB_NRS = XGB_NRS(df_A_house_train,df_A_house_test,predictors_A_house,target,params_house_A)    
Pred_B_H_XGB_NRS, Imp_B_H_XGB_NRS = XGB_NRS(df_B_house_train,df_B_house_test,predictors_B_house,target,params_house_B)    
Pred_C_H_XGB_NRS, Imp_C_H_XGB_NRS = XGB_NRS(df_C_house_train,df_C_house_test,predictors_C_house,target,params_house_C)    

PDCT_XGB_A_NRS = Imp_A_H_XGB_NRS[Imp_A_H_XGB_NRS['values'] >= Imp_A_H_XGB_NRS.quantile(0.4)['values']]['columns']             
Pred_A_H_XGB_NRS, testa = XGB_NRS(df_A_house_train,df_A_house_test,PDCT_XGB_A_NRS,target,params_house_A)    

#Not usefull after test
#PDCT_XGB_B_NRS = Imp_B_H_XGB_NRS[Imp_B_H_XGB_NRS['values'] >= Imp_B_H_XGB_NRS.quantile(0.3)['values']]['columns']             
#Pred_B_H_XGB_NRS, testb = XGB_NRS(df_B_house_train,df_B_house_test,PDCT_XGB_B_NRS,target,params_house_B)    

PDCT_XGB_C_NRS = Imp_C_H_XGB_NRS[Imp_C_H_XGB_NRS['values'] >= Imp_C_H_XGB_NRS.quantile(0.3)['values']]['columns']             
Pred_C_H_XGB_NRS, testc = XGB_NRS(df_C_house_train,df_C_house_test,PDCT_XGB_C_NRS,target,params_house_C)    

# In[] XGB TEST RESAMPLED DATASET   
Pred_A_H_XGB_RS, Imp_A_H_XGB_RS = XGB_RS(df_A_resampled_test,df_A_not_resampled,df_A_house_test,predictors_A_house,target,params_house_A)    
Pred_B_H_XGB_RS, Imp_B_H_XGB_RS = XGB_RS(df_B_resampled_test,df_B_not_resampled,df_B_house_test,predictors_B_house,target,params_house_B)    
Pred_C_H_XGB_RS, Imp_C_H_XGB_RS = XGB_RS(df_C_resampled_test,df_C_not_resampled,df_C_house_test,predictors_C_house,target,params_house_C)    

PDCT_XGB_A_RS = Imp_A_H_XGB_RS[Imp_A_H_XGB_RS['values'] >= Imp_A_H_XGB_RS.quantile(0.3)['values']]['columns']             
Pred_A_H_XGB_RS, testa = XGB_RS(df_A_resampled_test,df_A_not_resampled,df_A_house_test,PDCT_XGB_A_RS,target,params_house_A)    

#Not usefull after test
#PDCT_XGB_B_RS = Imp_B_H_XGB_RS[Imp_B_H_XGB_RS['values'] >= Imp_B_H_XGB_RS.quantile(0.3)['values']]['columns']             
#Pred_B_H_XGB_RS, testb = XGB_RS(df_B_resampled_test,df_B_not_resampled,df_B_house_test,predictors_modified_B,target,params_house_B)    

PDCT_XGB_C_RS = Imp_C_H_XGB_RS[Imp_C_H_XGB_RS['values'] >= Imp_C_H_XGB_RS.quantile(0.4)['values']]['columns']             
Pred_C_H_XGB_RS, testc = XGB_RS(df_C_resampled_test,df_C_not_resampled,df_C_house_test,PDCT_XGB_C_RS,target,params_house_C)    

# In[] XGB PRED NOT RESAMPLED    
Pred_A_H_XGB_NRS = XGB(df_A_house_train,df_A_house_test,predictors_A_house,target,params_house_A,202)    
Pred_B_H_XGB_NRS = XGB(df_B_house_train,df_B_house_test,predictors_B_house,target,params_house_B,138)    
Pred_C_H_XGB_NRS = XGB(df_C_house_train,df_C_house_test,predictors_C_house,target,params_house_C,133)    

# In[] XGB PRED RESAMPLED    
Pred_A_H_XGB_RS = XGB(df_A_resampled,df_A_house_test,predictors_A_house,target,params_house_A,238)    
Pred_B_H_XGB_RS = XGB(df_B_resampled,df_B_house_test,predictors_B_house,target,params_house_B,170)    
Pred_C_H_XGB_RS = XGB(df_C_resampled,df_C_house_test,predictors_C_house,target,params_house_C,62)    

## In[] XGB PRED NOT RESAMPLED - MODIFIED DATASETS
#Pred_A_H_XGB_NRS = XGB(df_A_house_train,df_A_house_test,PDCT_XGB_A_NRS,target,params_house_A,202)    
#Pred_B_H_XGB_NRS = XGB(df_B_house_train,df_B_house_test,predictors_B_house,target,params_house_B,138)    
#Pred_C_H_XGB_NRS = XGB(df_C_house_train,df_C_house_test,PDCT_XGB_C_NRS,target,params_house_C,133)    
#
## In[] XGB PRED RESAMPLED - MODIFIED DATASETS    
#Pred_A_H_XGB_RS = XGB(df_A_resampled,df_A_house_test,PDCT_XGB_A_RS,target,params_house_A,238)    
#Pred_B_H_XGB_RS = XGB(df_B_resampled,df_B_house_test,predictors_B_house,target,params_house_B,170)    
#Pred_C_H_XGB_RS = XGB(df_C_resampled,df_C_house_test,PDCT_XGB_C_RS,target,params_house_C,62)    

# In[]
##############################################################################################################
#                                               LIGHT GBM                                                    #
##############################################################################################################
params_gbm_a = {
    'learning_rate': 0.01,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 45,
    'max_depth': 5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 20
}
rounds_gbm_a = 6000
# In[]
params_gbm_b = {
    'learning_rate': 0.01,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 5,
    'max_depth': 3,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.95,
    'bagging_freq': 10
}
rounds_gbm_b = 700
# In[]
params_gbm_c = {
    'learning_rate': 0.01,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 5,
    'max_depth': 10,
    'feature_fraction': 0.95,
    'bagging_fraction': 0.95,
    'bagging_freq': 10
}
rounds_gbm_c = 6000

# In[]
def lightGBM_NRS(df_train,predictors,target,params,rounds):
    x_train, x_valid, y_train, y_valid = train_test_split(df_train[predictors], df_train[target], test_size=0.1, random_state=4242)

    d_train = lgb.Dataset(x_train[predictors],label=y_train) 
        
    bst = lgb.train(params, d_train, rounds)
    
    predictions = bst.predict(x_valid[predictors])
    
    ax = lgb.plot_importance(bst, max_num_features=len(bst.feature_name()))
    plt.show()
    
    importance = pd.DataFrame({'columns':bst.feature_name(),'values':bst.feature_importance()})
    importance = importance.sort_values(['values'], ascending = False)
    importance = InitializeIndex(importance)
    
    logloss = metrics.log_loss(y_true = y_valid,y_pred=predictions, eps=1e-15, normalize=True, sample_weight=None, labels=None)
    print("Log loss :",logloss)
    
    return(predictions,importance)
    
def lightGBM_RS(df_train_resampled,df_train_not_resampled,predictors,target,params,rounds):
    d_train = lgb.Dataset(df_train_resampled[predictors],label=df_train_resampled[target]) 
      
    bst = lgb.train(params, d_train, rounds)
    
    predictions = bst.predict(df_train_not_resampled[predictors])
    
    ax = lgb.plot_importance(bst, max_num_features=len(bst.feature_name()))
    plt.show()
    
    importance = pd.DataFrame({'columns':bst.feature_name(),'values':bst.feature_importance()})
    importance = importance.sort_values(['values'], ascending = False)
    importance = InitializeIndex(importance)
    
    logloss = metrics.log_loss(y_true = df_train_not_resampled[target],y_pred=predictions, eps=1e-15, normalize=True, sample_weight=None, labels=None)
    print("Log loss :",logloss)
    
    return(predictions, importance)
    
def lightGBM(df_train,df_test,predictors,target,params,rounds):
    
    d_train = lgb.Dataset(df_train[predictors],label=df_train[target]) 
    
    bst = lgb.train(params, d_train, rounds)
    predictions = bst.predict(df_test[predictors])
    return(predictions)    

# In[]    
Pred_A_H_GBM_NRS, IMP_A_H_GBM_NRS = lightGBM_NRS(df_A_house_train,predictors_A_house,target,params_gbm_a,rounds_gbm_a)    
Pred_B_H_GBM_NRS, IMP_B_H_GBM_NRS = lightGBM_NRS(df_B_house_train,predictors_B_house,target,params_gbm_b,rounds_gbm_a)    
Pred_C_H_GBM_NRS, IMP_C_H_GBM_NRS = lightGBM_NRS(df_C_house_train,predictors_C_house,target,params_gbm_c,rounds_gbm_a)    

PDCT_GBM_A_NRS = IMP_A_H_GBM_NRS[IMP_A_H_GBM_NRS['values'].astype('float64') >= IMP_A_H_GBM_NRS.quantile(0.3)['values']]['columns']             
Pred_A_H_GBM_NRS, testa = lightGBM_NRS(df_A_house_train,PDCT_GBM_A_NRS,target,params_gbm_a,rounds_gbm_a)       

PDCT_GBM_B_NRS = IMP_B_H_GBM_NRS[IMP_B_H_GBM_NRS['values'].astype('float64') >= IMP_B_H_GBM_NRS.quantile(0.6)['values']]['columns']             
Pred_B_H_GBM_NRS, testb = lightGBM_NRS(df_B_house_train,PDCT_GBM_B_NRS,target,params_gbm_b,rounds_gbm_b)     

PDCT_GBM_C_NRS = IMP_C_H_GBM_NRS[IMP_C_H_GBM_NRS['values'].astype('float64') >= IMP_C_H_GBM_NRS.quantile(0.4)['values']]['columns']             
Pred_C_H_GBM_NRS, testc = lightGBM_NRS(df_C_house_train,PDCT_GBM_C_NRS,target,params_gbm_c,rounds_gbm_c)    
# In[]    
Pred_A_H_GBM_RS, IMP_A_H_GBM_RS = lightGBM_RS(df_A_resampled_test,df_A_not_resampled,predictors_A_house,target,params_gbm_a,rounds_gbm_a)    
Pred_B_H_GBM_RS, IMP_B_H_GBM_RS = lightGBM_RS(df_B_resampled_test,df_B_not_resampled,predictors_B_house,target,params_gbm_b,rounds_gbm_b)    
Pred_C_H_GBM_RS, IMP_C_H_GBM_RS = lightGBM_RS(df_C_resampled_test,df_C_not_resampled,predictors_C_house,target,params_gbm_c,rounds_gbm_c)    

PDCT_GBM_A_RS = IMP_A_H_GBM_RS[IMP_A_H_GBM_RS['values'].astype('float64') >= IMP_A_H_GBM_RS.quantile(0.1)['values']]['columns']             
Pred_A_H_GBM_RS, testa = lightGBM_RS(df_A_resampled_test,df_A_not_resampled,PDCT_GBM_A_RS,target,params_gbm_a,rounds_gbm_a)       

PDCT_GBM_B_RS = IMP_B_H_GBM_RS[IMP_B_H_GBM_RS['values'].astype('float64') >= IMP_B_H_GBM_RS.quantile(0.8)['values']]['columns']             
Pred_B_H_GBM_RS, testb = lightGBM_RS(df_B_resampled_test,df_B_not_resampled,PDCT_GBM_B_RS,target,params_gbm_b,rounds_gbm_b)     

PDCT_GBM_C_RS = IMP_C_H_GBM_RS[IMP_C_H_GBM_RS['values'].astype('float64') >= IMP_C_H_GBM_RS.quantile(0.1)['values']]['columns']             
Pred_C_H_GBM_RS, testc = lightGBM_RS(df_C_resampled_test,df_C_not_resampled,PDCT_GBM_C_RS,target,params_gbm_c,rounds_gbm_c) 
# In[]    
Pred_A_H_GBM_NRS = lightGBM(df_A_house_train,df_A_house_test,predictors_A_house,target,params_gbm_a,rounds_gbm_a)    
Pred_B_H_GBM_NRS = lightGBM(df_B_house_train,df_B_house_test,predictors_B_house,target,params_gbm_b,rounds_gbm_b)    
Pred_C_H_GBM_NRS = lightGBM(df_C_house_train,df_C_house_test,predictors_C_house,target,params_gbm_c,rounds_gbm_c)    

# In[]    
Pred_A_H_GBM_RS = lightGBM(df_A_resampled,df_A_house_test,predictors_A_house,target,params_gbm_a,rounds_gbm_a)    
Pred_B_H_GBM_RS = lightGBM(df_B_resampled,df_B_house_test,predictors_B_house,target,params_gbm_b,rounds_gbm_b)    
Pred_C_H_GBM_RS = lightGBM(df_C_resampled,df_C_house_test,predictors_C_house,target,params_gbm_c,rounds_gbm_c)    

# In[]  MODIFIED DATASETS  
#Pred_A_H_GBM_NRS = lightGBM(df_A_house_train,df_A_house_test,PDCT_GBM_A_NRS,target,params_gbm_a,rounds_gbm_a)    
#Pred_B_H_GBM_NRS = lightGBM(df_B_house_train,df_B_house_test,PDCT_GBM_B_NRS,target,params_gbm_b,rounds_gbm_b)    
#Pred_C_H_GBM_NRS = lightGBM(df_C_house_train,df_C_house_test,PDCT_GBM_C_NRS,target,params_gbm_c,rounds_gbm_c)    
#
## In[]  MODIFIED DATASETS   
#Pred_A_H_GBM_RS = lightGBM(df_A_resampled,df_A_house_test,PDCT_GBM_A_RS,target,params_gbm_a,rounds_gbm_a)    
#Pred_B_H_GBM_RS = lightGBM(df_B_resampled,df_B_house_test,PDCT_GBM_B_RS,target,params_gbm_b,rounds_gbm_b)    
#Pred_C_H_GBM_RS = lightGBM(df_C_resampled,df_C_house_test,PDCT_GBM_C_RS,target,params_gbm_c,rounds_gbm_c)    

# In[]
##############################################################################################################
#                                               NEURAL NETWORKS                                              #
##############################################################################################################

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'
        
def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

# In[]

def NN_NRS(df_train,predictors,target,epochs,eta,alpha,country,standard_scaler):
    X_train, X_test, y_train, y_test = train_test_split(df_train[predictors], df_train[target], test_size=0.1, random_state=seed)
    
    #standard scaler
    if standard_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)  
        X_test = scaler.fit_transform(X_test)
   
    # create model
    model = Sequential()
    model.add(Dense(X_train.shape[1] + 1, input_dim=X_train.shape[1], activation='swish'))
    model.add(Dropout(0.1))
    model.add(Dense(round(len(df_train)/(alpha * (X_train.shape[1] + 1))), activation='swish'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    optimizer =optimizers.Adam(lr = eta)
    model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['binary_accuracy','accuracy'])

    #keep best weight
    filepath= path + country + "_NRS_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_test),np.array(y_test)),epochs = epochs,callbacks = callbacks_list)



def NN_RS(df_train_resampled,df_train_not_resampled,predictors,target,epochs,eta,alpha,country,standard_scaler):
    #standard scaler
    X_train = np.array(df_train_resampled[predictors])
    X_test = np.array(df_train_not_resampled[predictors])
    if standard_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(np.array(df_train_resampled[predictors]))  
        X_test = scaler.fit_transform(np.array(df_train_not_resampled[predictors]))
        
    # create model
    model = Sequential()
    model.add(Dense(df_train_resampled[predictors].shape[1] + 1, input_dim=df_train_resampled[predictors].shape[1],activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(round(len(df_train_resampled[predictors])/(alpha * (df_train_resampled[predictors].shape[1] + 1))), activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    optimizer = optimizers.Adam(lr = eta)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy','accuracy'])

    #keep best weight   
    filepath= path + country + "_RS_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(X_train, np.array(df_train_resampled[target]), validation_data=(X_test,np.array(df_train_not_resampled[target])),epochs = epochs,callbacks = callbacks_list)


#def NN(df_train,df_test,predictors,target,epochs,eta,alpha,weights):
#    # create model
#    model = Sequential()
#    model.add(Dense(df_train[predictors].shape[1] + 1, input_dim=df_train[predictors].shape[1],kernel_initializer='uniform', activation='swish',kernel_constraint=maxnorm(2)))
#    model.add(Dropout(0.2))
#    model.add(Dense(round(len(df_train[predictors])/(alpha * (df_train[predictors].shape[1] + 1))), activation='swish'))
#    model.add(Dropout(0.2))
#    model.add(Dense(1, activation='sigmoid'))
#    # Compile model
#    optimizer =optimizers.Adam(lr = eta)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy','accuracy'])
#    # Fit the model
#    model.fit(np.array(df_train[predictors]), np.array(df_train[target]),epochs = epochs)
#    predictions = model.predict(np.array(df_test[predictors]))
#    # Fit the model
#    return(predictions)
def NN(df_train,df_test,predictors,eta,alpha,weights,standard_scaler):
    X_test = np.array(df_test[predictors])
    
    # Fit the model
    if standard_scaler:
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)
        
    # create model
    model = Sequential()
    model.add(Dense(df_train[predictors].shape[1] + 1, input_dim=df_train[predictors].shape[1],activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(round(len(df_train[predictors])/(alpha * (df_train[predictors].shape[1] + 1))), activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.load_weights(weights)
    optimizer = optimizers.Adam(lr = eta)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy','accuracy'])

    predictions = model.predict(X_test)
    # Fit the model
    return(predictions)

def NN_bis(df_train,df_test,predictors,eta,alpha,weights,standard_scaler):
    X_test = np.array(df_test[predictors])
    
    # Fit the model
    if standard_scaler:
        scaler = StandardScaler()
        X_test = scaler.fit_transform(X_test)
        
    # create model
    model = Sequential()
    model.add(Dense(df_train[predictors].shape[1] + 1, input_dim=df_train[predictors].shape[1],activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.load_weights(weights)
    optimizer = optimizers.Adam(lr = eta)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy','accuracy'])

    predictions = model.predict(X_test)
    # Fit the model
    return(predictions)
    
# In[]:
# PLOT
#plt.clf()
#p.clf()
#import matplotlib.pyplot as p
#
#
#p.imshow(model.get_weights()[4], cmap='hot', interpolation='nearest')
#p.savefig(path + 'layer_4.png')

# In[]
#p.clf()
#plt.clf()
#import matplotlib.pyplot as plt
#plt.hist(model.get_weights()[1], bins='auto')  # arguments are passed to np.histogram
#plt.savefig(path + 'layer_1.png')
# In[]:
    
seed = randint(0,999)
np.random.seed(seed)
# In[]
NN_NRS(df_A_house_train,predictors_A_house,target,25,0.0001,2,"A",False)
NN_RS(df_A_resampled_test,df_A_not_resampled,predictors_A_house,target,25,0.0001,2,"A",False)    

# In[]
NN_NRS(df_B_house_train,predictors_B_house,target,25,0.002,8,"B",True)
NN_RS(df_B_resampled_test,df_B_not_resampled,predictors_B_house,target,25,0.002,8,"B",True)    

# In[]
NN_NRS(df_B_house_train,predictors_B_house,target,25,0.002,8,"B",True)
NN_RS(df_B_resampled_test,df_B_not_resampled,predictors_B_house,target,25,0.002,8,"B",True)    

# In[]
path_weights = path + 'weights_normal/'
Pred_A_H_NN_NRS = NN(df_A_house_train,df_A_house_test,predictors_A_house,0.0001,2,path_weights + 'A_NRS_weights_best.hdf5',False)
Pred_A_H_NN_RS = NN_bis(df_A_resampled,df_A_house_test,predictors_A_house,0.0001,2,path_weights + 'A_RS_weights_best.hdf5',False)

Pred_B_H_NN_NRS = NN(df_B_house_train,df_B_house_test,predictors_B_house,0.0001,8,path_weights + 'B_NRS_weights_best.hdf5',True)
Pred_B_H_NN_RS = NN(df_B_resampled,df_B_house_test,predictors_B_house,0.0001,8,path_weights + 'B_RS_weights_best.hdf5',True)











# In[]
##############################################################################################################
#                                               ENSEMBLING                                                   #
##############################################################################################################

Category_A_House = [pd.DataFrame(Pred_A_H_XGB_NRS),pd.DataFrame(Pred_A_H_XGB_RS),pd.DataFrame(Pred_A_H_GBM_NRS),pd.DataFrame(Pred_A_H_GBM_RS),pd.DataFrame(Pred_A_H_NN_NRS),pd.DataFrame(Pred_A_H_NN_RS)]
Category_B_House = [pd.DataFrame(Pred_B_H_XGB_NRS),pd.DataFrame(Pred_B_H_XGB_RS),pd.DataFrame(Pred_B_H_GBM_NRS),pd.DataFrame(Pred_B_H_GBM_RS)]
Category_C_House = [pd.DataFrame(Pred_C_H_XGB_NRS),pd.DataFrame(Pred_C_H_XGB_RS),pd.DataFrame(Pred_C_H_GBM_NRS),pd.DataFrame(Pred_C_H_GBM_RS)]

Category_A_House = pd.concat(Category_A_House,axis = 1)
Category_B_House = pd.concat(Category_B_House,axis = 1)
Category_C_House = pd.concat(Category_C_House,axis = 1)

Category_A_House.columns = ['A_XGB_NRS','A_XGB_RS','A_GBM_NRS','A_GBM_RS','A_NN_NRS','A_NN_RS']
Category_B_House.columns = ['B_XGB_NRS','B_XGB_RS','B_GBM_NRS','B_GBM_RS']
Category_C_House.columns = ['C_XGB_NRS','C_XGB_RS','C_GBM_NRS','C_GBM_RS']

pred_A_H = Category_A_House.mean(axis = 1)
pred_B_H = Category_B_House.mean(axis = 1)
pred_C_H = Category_C_House.mean(axis = 1)

# In[]
##############################################################################################################
#                                               SUBMISSION                                                   #
##############################################################################################################
result_A = pd.DataFrame({'id':df_A_house_test['id'],'country':'A','poor':pred_A_H})
result_B = pd.DataFrame({'id':df_B_house_test['id'],'country':'B','poor':pred_B_H})
result_C = pd.DataFrame({'id':df_C_house_test['id'],'country':'C','poor':pred_C_H})

# In[]
frames = [result_A,result_B,result_C]

submission = pd.concat(frames,axis = 0)
submission = submission[['id','country','poor']]
submission.to_csv(path + 'submission.csv',sep=',',index=False)
