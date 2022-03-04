# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:52:25 2019

Random Forest Regression Analysis using bootstrapping for average and std accuracy (R2 and RMSE)
Run the code in the same folder as your files

@author: Grazziela Figueredo
"""

import pandas as pd #for manipulating data
import numpy as np #for manipulating data

import sklearn #for building models
import sklearn.ensemble #for building models

from sklearn.model_selection import train_test_split #for creating a hold-out sample
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statistics

data = pd.read_excel('LigandSubstrateBoronDragonDescriptors_LASSO.xlsx')

# Determining X and y arrays. Y is supposed to be the last column of the input file
index = len(data.columns)
X = data.iloc[:,0:index-1]
y = data.iloc[:,index-1]

# Variable used to plot the y axis name in the regression graphs
dependent_variable = y.name

R2_train = list()
R2_test = list()
RMSE_train = list()
RMSE_test = list()

# Boostrapping
for cont in range(50):
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size = 0.3)

   # Random Forest regression 

   rf = sklearn.ensemble.RandomForestRegressor()
   rf.fit(X_train, y_train)
 
   # Training data
   y_hat = rf.predict(X_train)
   rmse = np.sqrt(mean_squared_error(y_train, y_hat))
   print("\nRMSE train RF: %.3f" % (rmse))
   coefficient_of_dermination = r2_score(y_train, y_hat)
   print("R2 train RF: %.3f" % (coefficient_of_dermination))
   R2_train.append(coefficient_of_dermination)
   RMSE_train.append(rmse)

   # Test data
   y_hat = rf.predict(X_test)
   rmse = np.sqrt(mean_squared_error(y_test, y_hat))
   print("RMSE test RF: %.3f" % (rmse))
   coefficient_of_dermination = r2_score(y_test, y_hat)
   print("R2 test RF: %.3f \n" % (coefficient_of_dermination))
   R2_test.append(coefficient_of_dermination)
   RMSE_test.append(rmse)
   
# Printing final stats for multiple runs   
   
print('Average R2 Training: %.3f' % (sum(R2_train) / len(R2_train)))
print('Standard Deviation R2 Training %.3f' % statistics.stdev(R2_train))

print('Average RMSE Training %.3f' % (sum(RMSE_train) / len(RMSE_train)))
print('Standard Deviation RMSE Training %.3f' % statistics.stdev(RMSE_train))

print('Average R2 Test %.3f' % (sum(R2_test) / len(R2_test)))
print('Standard Deviation R2 Test %.3f' % statistics.stdev(R2_test))

print('Average RMSE Test %.3f' % (sum(RMSE_test) / len(RMSE_test)))
print('Standard Deviation RMSE Test %.3f' % statistics.stdev(RMSE_test))
 
