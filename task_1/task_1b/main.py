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
    phi_6, phi_7, phi_8, phi_9, phi_10 = [], [], [], [], []                     #for quadratic features
    phi_11, phi_12, phi_13, phi_14, phi_15 = [], [], [], [], []                 #for exponential features
    phi_16, phi_17, phi_18, phi_19, phi_20 = [], [], [], [], []                 #for cosine features
    phi_21 = []

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

    X = [0] * len(X_lin)                                                            # I have to initialize X, otherwise I can't use np.c_
    if feature_index_vector[0] == 1:
        X = np.c_[X, X_lin]
    if feature_index_vector[1] == 1:
        X = np.c_[X, phi_6, phi_7, phi_8, phi_9, phi_10]
    if feature_index_vector[2] == 1:
        X = np.c_[X, phi_11, phi_12, phi_13, phi_14, phi_15]
    if feature_index_vector[3] == 1:
        X = np.c_[X, phi_16, phi_17, phi_18, phi_19, phi_20]
    if feature_index_vector[4] == 1:
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
number_of_final_features = 1
feature_index_vector = [1,1,1,1,1]

for j in range(5-number_of_final_features):
    rmse = []
    for i in range(5):
        if feature_index_vector[i] != 0:
            feature_index_vector[i] = 0
            print(feature_index_vector)
            X = generate_features(X_lin, feature_index_vector)
            w_star = ridge_regression(X, y, 100)
            rmse.append(root_mean_square_error(w_star, X, y))
            feature_index_vector[i] = 1
    plt.scatter(j,np.min(rmse))                                                 # Plot rmse
    feature_index_vector[np.argmax(rmse)] = 0
plt.show()
plt.close()

"""file = open("weights.csv", "w")                                              #Create submission file with writing access
for i in range(number_of_features):                                             #Write stuff to the submission file
    file.write(str(w_star[i]))
    file.write('\n')
file.close()"""
