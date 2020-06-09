import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.utils import shuffle

### PARAMETERS
number_of_iterations = 1

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data():

    read_train = pd.read_csv('../../data_1/data_1a/train.csv', delimiter=',')
    Id = read_train.iloc[:,0]
    y = read_train.iloc[:,1]
    X = read_train.iloc[:,2:]

    return [Id,y,X]

#DIVIDE THE DATA INTO DIFFERENT SECTIONS
def make_batches_forkwise(Id, number_of_batches):
    size = len(Id)                                                              #Get size of whole data
    batches = [[ 0 for i in range(0)] for i in range(number_of_batches)]        #Initialize empty frame for the batches

    for i in range(number_of_batches):                                          #Fill in the batches array with the Id, each one spread equally over the whole data
        for j in range(size):
            if Id[j]%number_of_batches==i:
                batches[i].append(Id[j])
    return batches

def make_batches_chronologically(Id, number_of_batches):
    batches = []
    for i in range(number_of_batches):
        batch_part = Id.iloc[int(len(Id)/10*(i)):int(len(Id)/10*(i+1))]
        batches.append(batch_part)
    return batches

def make_batches_randomly(Id, number_of_batches):
    batches = []
    for i in range(number_of_batches):
        batch_part = shuffle(Id, n_samples=int(len(Id)/10))
        Id = Id.drop(batch_part, axis=0)
        batches.append(batch_part)
    return batches

#EXACT SOLUTION OF THE RIDGE REGRESSION
def ridge_regression(X, y, alpha):
    w_star = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + alpha * np.identity(len(np.transpose(X)))), np.dot(np.transpose(X), y))
    return w_star


#------------------------------  MAIN  -----------------------------------------
[Id, y, X] = read_in_data()
number_of_batches = 10
batches = make_batches_chronologically(Id, number_of_batches)
#batches = make_batches_forkwise(Id, number_of_batches)

the_hack = []
for n_iteration in range(number_of_iterations):
    #batches = make_batches_randomly(Id, number_of_batches)

    rmse_mean = []
    for _lambda in [0.01, 0.1, 1, 10, 100]:                                         #For each lambda
        rmse = []
        for i in range(number_of_batches):
            X_TRAIN = X.drop(batches[i], axis=0)
            X_VALIDATION = X.iloc[batches[i],:]
            Y_LABELS = y.drop(batches[i], axis=0)
            Y_VALIDATION = y.iloc[batches[i]]
            #OPTION 1
            """model = Ridge(alpha=[_lambda])
            model.fit(X_TRAIN, Y_LABELS)
            y_pred = model.predict(X_VALIDATION)
            del model"""
            #OPTION 2
            w_star = ridge_regression(X_TRAIN, Y_LABELS, _lambda)                             #Compute the minimum weights
            y_pred = np.dot(np.transpose(w_star), np.transpose(X_VALIDATION))                   #Predict the output with this weights and the validation data

            rmse.append(np.sqrt(mean_squared_error(y_pred, Y_VALIDATION)))                 #And compute the rmse w.r.t. the validation data for all alpha
        rmse_mean.append(np.sum(rmse) / len(rmse))                                        #Compute the mean of the rmse values for all batches for a specific lambda
    print(rmse_mean)
    the_hack.append(rmse_mean)

the_hack_pd = pd.DataFrame(data=the_hack)
M_submission = the_hack_pd.mean()
print(M_submission)
M_submission.to_csv(r'alpha.csv', index=False, header=None)
