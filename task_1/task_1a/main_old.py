import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pandas as pd

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data():

    read_train = pd.read_csv('../../data_1/data_1a/train.csv', delimiter=',')
    Id = read_train.iloc[:,0]
    y = read_train.iloc[:,1]
    X = read_train.iloc[:,2:]

    return [Id,y,X]

#DIVIDE THE DATA INTO DIFFERENT SECTIONS
def make_batches(Id, number_of_batches):
    size = len(Id)                                                              #Get size of whole data
    batches = [[ 0 for i in range(0)] for i in range(number_of_batches)]        #Initialize empty frame for the batches

    for i in range(number_of_batches):                                          #Fill in the batches array with the Id, each one spread equally over the whole data
        for j in range(size):
            if Id[j]%number_of_batches==i:
                batches[i].append(Id[j])
    print(batches)
    return batches

#EXACT SOLUTION OF THE RIDGE REGRESSION
def ridge_regression(X, y, alpha):
    w_star = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + alpha * np.identity(len(np.transpose(X)))), np.dot(np.transpose(X), y))
    return w_star


#------------------------------  MAIN  -----------------------------------------
[Id, y, X] = read_in_data()
number_of_batches = 10
batches = make_batches(Id,number_of_batches)

file = open("alpha.csv", "w")                                                   #Create submission file with writing access

for _lambda in [0.01, 0.1, 1, 10, 100]:                                         #For each lambda
    rmse = []
    for i in range(number_of_batches):
        X_red = X[:]                                                            #Copy the whole matrix X to X_red
        y_red = y[:]                                                            #Copy the whole vector y to y_red
        X_val = []
        y_val = []
        for id in batches[i][::-1]:
            X_val.append(X[id])                                                 #Append the validation by the i-th part of batches
            y_val.append(y[id])
            del(X_red[id])
            del(y_red[id])
        w_star = ridge_regression(X_red, y_red, _lambda)                             #Compute the minimum weights
        y_pred = np.dot(np.transpose(w_star), np.transpose(X_val))                   #Predict the output with this weights and the validation data
        rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))                 #And compute the rmse w.r.t. the validation data for all alpha
    rmse_mean = np.sum(rmse) / len(rmse)                                        #Compute the mean of the rmse values for all batches for a specific lambda

    file.write(str(rmse_mean))                                                  #Write stuff to the submission file
    file.write('\n')
file.close()
