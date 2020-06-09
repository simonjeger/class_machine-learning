import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.model_selection import KFold
import numpy.random as random
np.random.seed(300)

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data(file):

    read_train = pd.read_csv('../../data_1/data_1a/' + file, delimiter=',')
    data_train = read_train.to_numpy()

    Id = []                                                                     #initialize the used arrays
    y = []
    X = []

    for row in data_train:                                                      #Assign "Id", "y" and "X"
        Id.append(int(row[0]))
        y.append(row[1])
        X.append(list(row[2:]))
    return [Id,y,X]

#EXACT SOLUTION OF THE RIDGE REGRESSION
def ridge_regression(X_train,X_val,y_train, alpha):
    clf = Ridge(alpha = alpha, fit_intercept = False)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    return y_pred


[Id, y, X] = read_in_data('train.csv')
number_of_batches = 10
batches = make_batches(Id,number_of_batches)

file = open("alpha.csv", "w")                                                   #Create submission file with writing access
kf = KFold(n_splits=number_of_batches)
X = np.asarray(X)                                         #Magic stuff to arrange array in nice matrix form:)
y = np.asarray(y)


for _lambda in [0.01, 0.1, 1, 10, 100]:                                         #For each lambda
    rmse = []
    for train_index, test_index in kf.split(X):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        y_pred = ridge_regression(X_train,X_val,y_train, _lambda)               #Compute the minimum weights
        rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))                 #And compute the rmse w.r.t. the validation data for all alpha
    rmse_mean = np.mean(rmse)                                                   #Compute the mean of the rmse values for all batches for a specific lambda

    file.write(str(rmse_mean))                                                  #Write stuff to the submission file
    file.write('\n')
file.close()
