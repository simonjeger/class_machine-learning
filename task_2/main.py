import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import pandas as pd

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_features_data():
    read_train_features = pd.read_csv('../../data_2/train_features.csv', delimiter=',')
    data_train_features = read_train_features.to_numpy()
    read_train_labels = pd.read_csv('../../data_2/train_labels.csv', delimiter=',')
    data_train_labels = read_train_labels.to_numpy()

    pid = []
    Y_LABELS = []
    X_FEATURES = []

    for row in data_train_features:
        X_FEATURES.append(list(row[2:]))                                                 #from Age (column C) to pH (column AK)
#2.)    Define how to deal with the data from all the 12 hours...


    for row in data_train_labels:
        pid.append(int(row[0]))
        Y_LABELS.append(row[1:])                                                #from LABEL_BaseExcess to LABEL_Heartrate

    return [pid, Y_LABELS, X_FEATURES]




###---------------MAIN----------------------------------------------------------
[pid, Y_LABELS, X_FEATURES] = read_in_data()                                    #Read the data from features file

##Subtask 1:
#1.)    Define how to deal with different starting hours...


#3.)    Define how to deal with NaN number in the training set... (maybe use mean values?)


#4.)    Setting up a model with model = svm.SVC(kernel == 'linear')
        #model.fit(X, LABEL_i)

#5.)    Repeat training a model for every single features


#6.)    Writing a function for predicting the values for the Labels other than the default one. It should include the activation function \sigma from the exercise description


#7.)    Predicting the values for the differents tests ordered


#8.)    Writing the predictions to the sample.csv file


##Subtask 2:
#9.)    Train a model using the label sepsis (0 for no sepsis, 1 otherwise)


#10.)   Predicting the occurance of sepsis with the standard function model.predict
        #example:     if(model.predict([[X_FEATURES[i](Age), ..., X_FEATURES[i](pH) ]]))==0:


#11.)   Writing the prediction for sepsis to the sample.csv file





file = open("sample.csv", "w")                                                  #Create submission file with writing access



for i in range(number_of_features):                                             #Write stuff to the submission file
    file.write(str(w_star[i]))
    file.write('\n')
file.close()
