# Copyright 2022 Benjamin Owen, Apache 2.0
# University of Nottingham, GSK-EPSRC Prosperity Partnership EP/S035990/1
# and EPSRC award EP/S022236/1 

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold



def choose_model(best_params):
    if best_params == None:
        #return LinearRegression()
        #return RandomForestRegressor()
        return GradientBoostingRegressor()


    else:
        #return LinearRegression()
        #return RandomForestRegressor(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"], min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'])
        return GradientBoostingRegressor(loss = best_params['loss'], learning_rate=best_params['learning_rate'],n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'])
 


def choose_dataset():
    #return 'Cyclic'
    #return 'Acyclic'
    return 'Combined'
    #return 'Cyclohexenone'


def hyperparam_tune(X, y, model):
    print(str(model))
    if str(model) == 'RandomForestRegressor()':
        hyperP = dict(n_estimators=[100, 300, 500, 800], max_depth=[None, 5, 8, 15, 25, 30],
                      min_samples_split=[2, 5, 10, 15, 100],
                      min_samples_leaf=[1, 2, 5, 10])

    elif str(model) == 'GradientBoostingRegressor()':
        hyperP = dict(loss=['ls'], learning_rate=[0.1, 0.2, 0.3],
                      n_estimators=[100, 300, 500, 800], max_depth=[None, 5, 8, 15, 25, 30],
                      min_samples_split=[2],
                      min_samples_leaf=[1, 2])

    gridF = GridSearchCV(model, hyperP, cv=3, verbose=1, n_jobs=-1)
    bestP = gridF.fit(X, y)
    print(bestP.best_params_)
    return bestP.best_params_

random_seeds = np.random.random_integers(0, high=1000, size=30)
print(random_seeds)

descriptors = ['LVR1', 'LVR2', 'LVR3', 'LVR4', 'LVR5', 'LVR6', 'LVR7', 'VB', 'ER1', 'ER2', 'ER3', 'ER4', 'ER5', 'ER6',
               'ER7', 'SStoutR1', 'SStoutR2', 'SStoutR3', 'SStoutR4', '%top']

data = pd.read_csv(choose_dataset()+'.csv')

data = data.filter(descriptors)

#remove erroneous data
data = data.dropna(axis=0)


X = data.drop(['%top'], axis = 1)
X = RobustScaler().fit_transform(np.array(X))
y = data['%top']
print(X)
print(y)
best_params = hyperparam_tune(X, y, choose_model(best_params=None)) #hyperparameter tuning completed on whole subset
#best_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
#best_params = {'learning_rate': 0.2, 'loss': 'ls', 'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300}

r2_cv_scores = []
rmse_cv_scores = []
r2_val_scores = []
rmse_val_scores = []

for i in range(len(random_seeds)):
    # split into training and validation sets, 9:1
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.1, random_state=random_seeds[i])

    X_train = np.array(X_train).astype('float64')
    X_val = np.array(X_val).astype('float64')
    y_train = np.array(y_train).astype('float64')
    y_val = np.array(y_val).astype('float64')

    # 5 fold CV on training set, repeated 3 times
    for j in range(3):
        kfold = KFold(n_splits=5)
        for train, test in kfold.split(X_train, y_train):
            model = choose_model(best_params)
            model.fit(X_train[train], y_train[train])
            predictions = model.predict(X_train[test]).reshape(1, -1)[0]

            r2 = r2_score(y_train[test], predictions)
            rmse = math.sqrt(mean_squared_error(y_train[test], predictions))
            r2_cv_scores.append(r2)
            rmse_cv_scores.append(rmse)


    # predict on validaiton set
    model = choose_model(best_params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    r2 = r2_score(y_val, predictions)
    rmse = math.sqrt(mean_squared_error(y_val, predictions))
    r2_val_scores.append(r2)
    rmse_val_scores.append(rmse)


print('Model:',  model)
print('Data Subset: ',  choose_dataset())
print('Random Seeds: ', random_seeds, '\n')
print('Num CV Scores: ', len(r2_cv_scores))
print('CV R2 Mean: ', np.mean(np.array(r2_cv_scores)), '+/-', np.std(np.array(r2_cv_scores)))
print('CV RMSE Mean %: ', np.mean(np.array(rmse_cv_scores)), '+/-', np.std(np.array(rmse_cv_scores)), '\n')
print('Num Val Scores: ', len(r2_val_scores))
print(r2_val_scores)
print('Val r2 Mean: ', np.mean(np.array(r2_val_scores)), '+/-', np.std(np.array(r2_val_scores)))
print('Val RMSE Mean %: ', np.mean(np.array(rmse_val_scores)), '+/-', np.std(np.array(rmse_val_scores)))
