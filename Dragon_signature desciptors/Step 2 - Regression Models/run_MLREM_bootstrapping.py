import pandas as pd
import numpy as np
import sklearn.feature_selection 
import sklearn.preprocessing
import sklearn.model_selection
import mlr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statistics

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
    return np.sqrt(mean_squared_error(yt, yp))

def run_MLREM(df2, name,  dependent_variable, up_to_beta=200, screen_variance=False):

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
    x_remcorr = remove_correlation(x_pastvar,0.9)
     
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
        X_train2 = df_X_train[indice[1:]]
        
        # RMSE calculation - test set
        yp = np.dot(X_test2,w[1:])
        yp = y_scaller.inverse_transform(yp)
        yt = y_scaller.inverse_transform(y_test)
        sepai.append(sep(yp,yt))
        
        # RMSE calculation - training set
        yp = np.dot(X_train2,w[1:])
        yp = y_scaller.inverse_transform(yp)
        yt = y_scaller.inverse_transform(y_train)
        
        #print(beta, ';', sep(yp,yt),';', sepai[-1])
    

    
    # Extracting best results obtained in the previous loop based on the minimum error of prediction
    best_beta_indx = sepai.index(np.array(sepai).min())
    print('Best beta =', betai[best_beta_indx])
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
    y_train = y
    # stats calculation
    r2_train = r2_score(y,yp)
    rmse_train = np.sqrt(mean_squared_error(yp, y))
    print("\n\n R2 train MLREM: %f" % (r2_train))
    print("\n\n RMSE train MLREM: %f" % (rmse_train))
    
    # stats for the test set
    yp = np.dot(df_X_test[indexai[best_beta_indx]],weights[best_beta_indx]) 
    yp = y_scaller.inverse_transform(yp)
    y = y_scaller.inverse_transform(y_test)
    y_test = y
        
    r2_test = r2_score(y,yp)#stats.pearsonr(yp,y)[0]**2
    
    rmse_test = np.sqrt(mean_squared_error(yp, y))
    print("\n\n RMSE test MLREM: %f" % (rmse_test))
    
       
   
    return (rmse_train,rmse_test,r2_train,r2_test)


# Add here the csv file to be analysed and the name of the dependent variable
R2_train = list()
R2_test = list()
RMSE_train = list()
RMSE_test = list()

data = pd.read_excel('LigandsSubstrateBoronFragmentDescriptors_LASSO.xlsx')
dependent_variable = 'top'

for cont in range(50):
   (rmse_train,rmse_test,r2_train,r2_test) = run_MLREM(data,'', dependent_variable)
   RMSE_train.append(rmse_train)
   RMSE_test.append(rmse_test)
   R2_train.append(r2_train)
   R2_test.append(r2_test)
   
   
print('Average R2 Training: %.3f' % (sum(R2_train) / len(R2_train)))
print('Standard Deviation R2 Training %.3f' % statistics.stdev(R2_train))

print('Average RMSE Training %.3f' % (sum(RMSE_train) / len(RMSE_train)))
print('Standard Deviation RMSE Training %.3f' % statistics.stdev(RMSE_train))

print('Average R2 Test %.3f' % (sum(R2_test) / len(R2_test)))
print('Standard Deviation R2 Test %.3f' % statistics.stdev(R2_test))

print('Average RMSE Test %.3f' % (sum(RMSE_test) / len(RMSE_test)))
print('Standard Deviation RMSE Test %.3f' % statistics.stdev(RMSE_test))
 


#X_test.to_excel('xtest.xlsx')
#X_train.to_excel('xtrain.xlsx')
