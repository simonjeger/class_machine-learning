import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data(file):

    read_train = pd.read_csv('../../data_1/data_1b/' + file, delimiter=',')
    data_train = read_train.to_numpy()

    Id = []                                                                     #initialize the used arrays
    y = []
    X_lin = []

    for row in data_train:                                                      #Assign "Id", "y" and "X"
        Id.append(int(row[0]))
        y.append(row[1])
        X_lin.append(list(row[2:]))
    return [Id,y,X_lin]

def add_features(X_lin):
    #The matrix X must have as many columns as number_of_features
    phi_6, phi_7, phi_8, phi_9, phi_10 = [], [], [], [], []                     #for quadratic features
    phi_11, phi_12, phi_13, phi_14, phi_15 = [], [], [], [], []                 #for exponential features
    phi_16, phi_17, phi_18, phi_19, phi_20 = [], [], [], [], []                 #for cosine features
    phi_21 = []                                                                 #for constant features


    for row in X_lin:
        phi_6.append(row[0]**2)
        phi_7.append(row[1]**2)
        phi_8.append(row[2]**2)
        phi_9.append(row[3]**2)
        phi_10.append(row[4]**2)
        phi_11.append(np.exp(row[0]))
        phi_12.append(np.exp(row[1]))
        phi_13.append(np.exp(row[2]))
        phi_14.append(np.exp(row[3]))
        phi_15.append(np.exp(row[4]))
        phi_16.append(np.cos(row[0]))
        phi_17.append(np.cos(row[1]))
        phi_18.append(np.cos(row[2]))
        phi_19.append(np.cos(row[3]))
        phi_20.append(np.cos(row[4]))
        phi_21.append(1)

    X = np.c_[X_lin, phi_6, phi_7, phi_8, phi_9, phi_10, phi_11, phi_12, phi_13, phi_14, phi_15, phi_16, phi_17, phi_18, phi_19, phi_20, phi_21]
    return X

def generate_features(X_lin, feature_index_vector):
    # Example of feature_index_vector: [1,1,0,1,1] = linear, quadratic, no exponential, cosine , constant

    #The matrix X must have as many columns as number_of_features
    phi_1, phi_2, phi_3, phi_4, phi_5 = [], [], [], [], []
    phi_6, phi_7, phi_8, phi_9, phi_10 = [], [], [], [], []                     #for quadratic features
    phi_11, phi_12, phi_13, phi_14, phi_15 = [], [], [], [], []                 #for exponential features
    phi_16, phi_17, phi_18, phi_19, phi_20 = [], [], [], [], []                 #for cosine features
    phi_21 = []

    for row in X_lin:
        phi_1.append(row[0])
        phi_2.append(row[0])
        phi_3.append(row[0])
        phi_4.append(row[0])
        phi_5.append(row[0])
        phi_6.append(row[0]**2)
        phi_7.append(row[1]**2)
        phi_8.append(row[2]**2)
        phi_9.append(row[3]**2)
        phi_10.append(row[4]**2)
        phi_11.append(np.exp(row[0]))
        phi_12.append(np.exp(row[1]))
        phi_13.append(np.exp(row[2]))
        phi_14.append(np.exp(row[3]))
        phi_15.append(np.exp(row[4]))
        phi_16.append(np.cos(row[0]))
        phi_17.append(np.cos(row[1]))
        phi_18.append(np.cos(row[2]))
        phi_19.append(np.cos(row[3]))
        phi_20.append(np.cos(row[4]))
        phi_21.append(1)

    X = [0] * len(X_lin)
    if feature_index_vector[0] == 1:
        X = np.c_[X, phi_1]
    if feature_index_vector[1] == 1:
        X = np.c_[X, phi_2]
    if feature_index_vector[2] == 1:
        X = np.c_[X, phi_3]
    if feature_index_vector[3] == 1:
        X = np.c_[X, phi_4]
    if feature_index_vector[4] == 1:
        X = np.c_[X, phi_5]
    if feature_index_vector[5] == 1:
        X = np.c_[X, phi_6]
    if feature_index_vector[6] == 1:
        X = np.c_[X, phi_7]
    if feature_index_vector[7] == 1:
        X = np.c_[X, phi_8]
    if feature_index_vector[8] == 1:
        X = np.c_[X, phi_9]
    if feature_index_vector[9] == 1:
        X = np.c_[X, phi_10]
    if feature_index_vector[10] == 1:
        X = np.c_[X, phi_11]
    if feature_index_vector[11] == 1:
        X = np.c_[X, phi_12]
    if feature_index_vector[12] == 1:
        X = np.c_[X, phi_13]
    if feature_index_vector[13] == 1:
        X = np.c_[X, phi_14]
    if feature_index_vector[14] == 1:
        X = np.c_[X, phi_15]
    if feature_index_vector[15] == 1:
        X = np.c_[X, phi_16]
    if feature_index_vector[16] == 1:
        X = np.c_[X, phi_17]
    if feature_index_vector[17] == 1:
        X = np.c_[X, phi_18]
    if feature_index_vector[18] == 1:
        X = np.c_[X, phi_19]
    if feature_index_vector[19] == 1:
        X = np.c_[X, phi_20]
    if feature_index_vector[20] == 1:
        X = np.c_[X, phi_21]
    X = np.delete(X, 0, 1)                                                         # I delete the first column containing the zeros I added above
    return X

#EXACT SOLUTION OF THE LINEAR REGRESSION
def ridge_regression(X, y, _lambda):
    w_star = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + _lambda * np.identity(len(np.transpose(X)))), np.dot(np.transpose(X), y).T)
    #w_star = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))     #for linear regression (_lambda = 0)
    return w_star

#RMSE
def root_mean_square_error(w_star, X, y):
    y_pred = np.dot(np.transpose(w_star), np.transpose(X))
    return np.sqrt(mean_squared_error(y, y_pred))

###MAIN-------------------------------------------------------------------------
[Id, y, X_lin] = read_in_data('train.csv')                                      #Read the data from file
number_of_features = 21
number_of_final_features = 5
feature_index_vector = [1]*number_of_features

j = 0
rmse_min = []
while np.sum(feature_index_vector) > number_of_final_features:
    rmse = [-1]*len(feature_index_vector)
    for i in range(len(feature_index_vector)):
        if feature_index_vector[i] != 0:
            feature_index_vector[i] = 0
            X = generate_features(X_lin, feature_index_vector)
            w_star = ridge_regression(X, y, 0.0001)
            rmse[i] = (root_mean_square_error(w_star, X, y))
            feature_index_vector[i] = 1
    rmse_min.append(np.min([n for n in rmse  if n>0]))
    feature_index_vector[np.argmax(rmse)] = 0
plt.scatter(np.linspace(0,len(rmse_min), len(rmse_min)), rmse_min)
plt.show()
plt.close()

w_write = feature_index_vector
j = 0
for i in range(number_of_features):
    if feature_index_vector[i] == 1:
        w_write[i] = w_star[j]
        j = j+1

file = open("weights.csv", "w")                                              #Create submission file with writing access
for i in range(number_of_features):                                             #Write stuff to the submission file
    file.write(str(w_write[i]))
    file.write('\n')
file.close()
