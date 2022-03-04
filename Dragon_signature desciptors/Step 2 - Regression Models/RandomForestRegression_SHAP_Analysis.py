# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:52:25 2019

SHAP importance calculation for Random Forests with standard parameters
using training/test split

Obs: Run the code in the same folder as your data files

@author: Grazziela Figueredo
"""
import pandas as pd #for manipulating data
import numpy as np #for manipulating data

import sklearn #for building models
import sklearn.ensemble #for building models
from sklearn.model_selection import train_test_split #for creating a hold-out sample
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import shap #SHAP package for model interpretability
import matplotlib.pyplot as plt 
from matplotlib import cm

def plot_regression(y, y_hat, figure_title):
    fig, ax = plt.subplots()
    ax.scatter(y, y_hat)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured ' + dependent_variable, fontsize = 13)
    ax.set_ylabel('Predicted ' + dependent_variable, fontsize = 13)
    plt.title(figure_title, fontsize = 13)
    coefficient_of_dermination = r2_score(y, y_hat)
    legend = 'R2: '+str(float("{0:.2f}".format(coefficient_of_dermination)))
    plt.legend(['Best fit',legend],loc = 'upper left', fontsize = 13)
    plt.show()
    
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    print("\n\n RMSE train RF: %f" % (rmse)) 
    print("\n R2 train RF: %f" % (coefficient_of_dermination))


# Random Forest Regression using standard parameters
def random_forest_regression(X_train, y_train, X_test, y_test): 
   rf = sklearn.ensemble.RandomForestRegressor()
   rf.fit(X_train, y_train)
   y_hat = rf.predict(X_train)
 
   plot_regression(y_train, y_hat, "Results for the Training Set")
   y_hat = rf.predict(X_test)
   plot_regression(y_test, y_hat, "Results for the Test Set")
   
   return rf


# Reading input data
data = pd.read_excel('LigandSubstrateBoronDragonDescriptors_LASSO.xlsx')

# Determining X and y arrays. Y is supposed to be the last column of the input file
index = len(data.columns)
X = data.iloc[:,0:index-1]
y = data.iloc[:,index-1]

# Variable used to plot the y axis name in the regression graphs
dependent_variable = y.name

# Training and test sets split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size = 0.3)

##############################################################################

rf = random_forest_regression(X_train, y_train, X_test, y_test)

# Random Forest explainer
explainerRF = shap.TreeExplainer(rf)
shap_values_RF_test = explainerRF.shap_values(X_test)
shap_values_RF_train = explainerRF.shap_values(X_train)

df_shap_RF_test = pd.DataFrame(shap_values_RF_test, columns=X_test.columns.values)
df_shap_RF_train = pd.DataFrame(shap_values_RF_train, columns=X_train.columns.values)

# if a feature has 10 or less unique values then treat it as categorical
categorical_features = np.argwhere(np.array([len(set(X_train.values[:,x]))
for x in range(X_train.values.shape[1])]) <= 10).flatten()

# Printing SHAP results
print('Shap for RF:\n\n')
plt.figure()
shap.summary_plot(shap_values_RF_train, X_train, plot_type="bar", max_display = 8) 

plt.figure()
shap.summary_plot(shap_values_RF_train, X_train, max_display = 8, color_bar_label = 'Descriptor value', show = False, plot_size= (4.5,3))
plt.grid()
#Changing plot colours
for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if hasattr(fcc, "set_cmap"):
            fcc.set_cmap(cm.get_cmap('coolwarm'))
