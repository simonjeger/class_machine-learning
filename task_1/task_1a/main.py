import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pandas as pd

def read_in_data(file):

    #read data from csv file and transfer to numpy array
    read_train = pd.read_csv('../../data_1/data_1a/' + file, delimiter=',')  # read in train.csv file
    data_train = read_train.to_numpy()

    #initialize the used arrays
    Id = []
    y = []
    X = []

    for row in data_train:
        Id.append(int(row[0]))
        y.append(row[1])
        X.append(list(row[2:]))

    return [Id,y,X]

def make_batches(Id, number_of_batches):
    size = len(Id)
    batches = [[ 0 for i in range(0)] for i in range(number_of_batches)]

    for i in range(number_of_batches):
        for j in range(size):
            if Id[j]%number_of_batches==i:
                batches[i].append(Id[j])

    return batches

def ridge_regression(X, y, alpha):
    w_star = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + alpha * np.identity(len(np.transpose(X)))), np.dot(np.transpose(X), y))
    return w_star


#The file value is either "train.csv" or "test.csv"
[Id, y, X] = read_in_data('train.csv')
number_of_batches = 10
batches = make_batches(Id,number_of_batches)

for alpha in [0.01, 0.1, 1, 10, 100]:
    for i in range(number_of_batches):
        X_red = X
        y_red = y
        X_val = []
        y_val = []
        for id in batches[i][::-1]:
            X_val.append(X[id])
            y_val.append(y[id])
            del(X_red[id])
            del(y_red[id])
        w = ridge_regression(X_red, y_red, alpha)
        y_pred = np.dot(np.transpose(w), np.transpose(X_val))
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
