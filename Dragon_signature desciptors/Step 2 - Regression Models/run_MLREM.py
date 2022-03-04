# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:48:52 2019

@author: Paulius and adapted by Grazziela

This code was extracted from Paulius', but is largely uncommented, so there are 
a few parts I dont understand. Tried to comment it a bit more.
"""
import pandas as pd
import numpy as np
import sklearn.feature_selection 
import sklearn.preprocessing
import sklearn.model_selection
import mlr
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# sorting variables
def sort_by_feature_name(df):

    df =df.T
    a = []
    for i in df.T.columns:
        a.append(len(i))
    df["len"] = a
    df_sorted = df.sort_values(["len"])
    df_sorted = df_sorted.drop(["len"],axis=1)
    return df_sorted.T  

# Remove feature correlations, using Pearson correlation, based on the variable threshold
def remove_correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] >= threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset
    
# SEP is the standard error of prediction (test set). SEE is the error for training
def sep(yt,yp):
    return np.sqrt(np.mean((yt-yp)**2))

def run_MLREM(df2, name,  dependent_variable, up_to_beta=40, screen_variance=False):

    df=df2.copy()
    
    # Separating independent and dependent variables x and y
    y = df[dependent_variable].to_numpy().reshape(-1,1)
    x = df.drop(dependent_variable,axis=1)
    x_sorted=sort_by_feature_name(x)
    x_pastvar = x_sorted.copy()
    
    if screen_variance:
        selector = sklearn.feature_selection.VarianceThreshold(threshold=0.01)
        selector.fit(x_sorted)
        x_pastvar=x_sorted.T[selector.get_support()].T
    x_remcorr = remove_correlation(x_pastvar,1)
     
    y_scaller = sklearn.preprocessing.StandardScaler()
    x_scaller = sklearn.preprocessing.StandardScaler()
    
    ys_scalled = y_scaller.fit_transform(y)
    xs_scalled = x_scaller.fit_transform(x_remcorr)
    ind = x_remcorr.columns
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(xs_scalled, ys_scalled, test_size=0.3)
    
    df_X_test = pd.DataFrame(X_test, columns=ind) # this is to be able to calculate SEP for each iteration of beta
    df_X_train = pd.DataFrame(X_train, columns=ind)
    
    sepai = []
    betai = []
    indexai = []
    weights = []
    pvalues = []
    
    # Beta optimisation
    for i in range(1,up_to_beta):
        beta = 0.1 * i
        betai.append(beta)
        w, indice, pv = mlr.train(X_train, y_train, ind,  beta=beta)
        indexai.append(indice)
        weights.append(w)
        pvalues.append(pv)
        
        X_test2 = df_X_test[indice[1:]]

        sepai.append(sep(np.dot(X_test2,w[1:]),y_test))
    
    # Extracting best results obtained in the previous loop based on the minimum error of prediction
    best_beta_indx = sepai.index(np.array(sepai).min())

    # weights for each remaining feature after correlation has been performed
    df_features = pd.DataFrame(weights[best_beta_indx],index=indexai[best_beta_indx])
    df_features.columns = ["weights"]
    # p value calculation for the regression
    df_pvalues = pd.DataFrame(pvalues[best_beta_indx],index=indexai[best_beta_indx])
    df_pvalues.columns = ["pvalues"]

    # Indexes of the features selected for regression after correlation study is performed
    saved_idx_MLR = df_features.index.tolist() 
    saved_idx_MLR.append(dependent_variable)
    
    # Intercept is a feature used in the MLR function
    if 'Intercept' in saved_idx_MLR:
        df_X_train['Intercept'] = 1
        df_X_test['Intercept'] = 1
        df['Intercept'] = 1

    mlr_name = "DataForAnalysis_features_not_correlated"+ name +".csv"
    df[saved_idx_MLR].to_csv(mlr_name,sep=",",header=True)
    
    # stats for the training set    
    yp = np.dot(df_X_train[indexai[best_beta_indx]],weights[best_beta_indx]) 
    yp = y_scaller.inverse_transform(yp)
    y = y_scaller.inverse_transform(y_train)
    
    # stats calculation
    r2_train = r2_score(yp,y)
    see = sep(yp,y)
    print("\n\n  SEE: %f" % (see))

    
    # stats for the test set
    yp = np.dot(df_X_test[indexai[best_beta_indx]],weights[best_beta_indx]) 
    yp = y_scaller.inverse_transform(yp)
    y = y_scaller.inverse_transform(y_test)
        
    r2_test = r2_score(yp,y)
    sep2 = sep(yp,y)
    best_beta = betai[best_beta_indx]

    print("\n\n  SEP: %f" % (sep2))
    
    rmse = np.sqrt(mean_squared_error(yp, y))
    print("\n\n RMSE test: %f" % (rmse))
    
    # Plotting test set
    fig, ax = plt.subplots()
    ax.scatter(y, yp)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured ' + dependent_variable, fontsize = 13)
    ax.set_ylabel('Predicted ' + dependent_variable, fontsize = 13)
    figure_title = "Prediction Error for Multiple Linear Regression"
    plt.title(figure_title, fontsize = 13)
    legend = 'R2: '+str(float("{0:.2f}".format(r2_test)))
    plt.legend(['Best fit',legend],loc = 'upper left', fontsize = 13)
    plt.grid(axis='both')
    plt.show()
    
    
    metrics = "_r2_{:3.2f}_see_{:3.2f}_q2_{:3.2f}_sep2_{:3.2f}_beta_{:3.2f}".format(r2_train,see,r2_test,sep2,best_beta)
    
    file_stats = "StatsOutput_"+ name + metrics +".csv"
   
    df_stats = pd.concat([df_features,df_pvalues], axis=1)
    df_stats.to_csv(file_stats,sep=",",header=True)
    
    # plotting bar chart of feature contribution
    #del df_stats['Intercept']
    df_stats = df_stats.sort_values(by = ['weights'])
    df_stats['positive'] = df_stats['weights'] > 0
    df_stats['weights'].plot(kind='barh', grid = True,figsize=(5,4),
                             color=df_stats.positive.map({True: 'b', False: 'r'}))


    df_to_return = df[saved_idx_MLR]
    return df_to_return


# Add here the csv file to be analysed and the name of the dependent variable
data = pd.read_excel('LigandSubstrateBoronDragonDescriptors_LASSO.xlsx')
dependent_variable = 'top'
run_MLREM(data, 'CP', dependent_variable)


