import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from scipy.special import expit
import pandas as pd
import inspect

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data():
    #read train features (X_FEATURES)
    read_train_features = pd.read_csv('../data_2/train_features.csv', delimiter=',', nrows=120)
    read_train_features = read_train_features.replace('nan', np.NaN)
    #read_train_features.fillna(read_train_features.mean(), inplace=True)        #Dealing with "nan" values --> replacing them with the mean value of the whole column
    data_train_features = read_train_features.to_numpy()

    X_FEATURES = []
    for row in data_train_features:                                             #from Age (column C) to pH (column AK)
        X_FEATURES.append(list(row[2:]))

    #read train labels (pid and Y_LABELS)
    read_train_labels = pd.read_csv('../data_2/train_labels.csv', delimiter=',', nrows=10)
    data_train_labels = read_train_labels.to_numpy()

    pid = []
    Y_LABELS = []
    for row in data_train_labels:
        pid.append(int(row[0]))
        Y_LABELS.append(list(row[1:11]))

    #read test data (X_TEST)
    read_test_features = pd.read_csv('../data_2/test_features.csv', delimiter=',', nrows=120)
    read_test_features = read_test_features.replace('nan', np.NaN)
    read_test_features.fillna(read_test_features.mean(), inplace=True)          #Dealing with "nan" values --> replacing them with the mean value of the whole column
    data_test_features = read_test_features.to_numpy()

    X_TEST = []
    for row in data_test_features:
        X_TEST.append(list(row[2:]))

    return [pid, Y_LABELS, X_FEATURES, X_TEST]

def pre_processing(pid, X_LONG):
    mean = np.mean(X_LONG, axis=0)                                              #array with mean of all the values in the big matrix
    print(mean)
    std = np.std(X_LONG, axis=0)
    print(std)
    X_CUT = []
    for i in range(1,len(pid)+1):                                               #going from 1 through all patients
        X_Patient = X_LONG[(1+12*(i-1)):(12*i)]                                 #extract only the relevant data for that specific patient
        X_append = np.mean(X_Patient, axis=0)
        for j in range(0,len(X_append)):
            if np.isnan(X_append[j]):
                X_append[j] = np.random.normal(mean, std, 1)
        X_CUT.append(list(X_append))                                                  #axis=0 for taking the mean value over columns instead of rows


    return X_CUT

def predict(X_TEST_cut):            #TODO!!!
    return expit(X_TEST_cut)


###---------------MAIN----------------------------------------------------------
[pid, Y_LABELS, X_FEATURES, X_TEST] = read_in_data()                            #Read the data from features file

#nans have been replaced and now the X_FEATURES is shrinked to only one row per patient using mean approximation
X_TRAIN = pre_processing(pid, X_FEATURES)                                       #from now on work with X_DATA

##Subtask 1: Setting up a model with multiclass labels
model = OneVsRestClassifier(svm.SVC(kernel='linear'))
#print(X_DATA)
model.fit(X_TRAIN, Y_LABELS)

X_VALID = pre_processing(pid, X_TEST)
#Prediction
y_pred = model.predict(X_VALID)
print(y_pred)
#y_mypred = predict(X_VALID)
#print(y_mypred)

#for i in range(0,n_cutoff):
#    print(y_pred_cut[12*i])




##Subtask 2:
#9.)    Train a model using the label sepsis (0 for no sepsis, 1 otherwise)


#10.)   Predicting the occurance of sepsis with the standard function model.predict
        #example:     if(model.predict([[X_FEATURES[i](Age), ..., X_FEATURES[i](pH) ]]))==0:


#11.)   Writing the LABELS and the prediction for sepsis to the sample.csv file





#file = open("sample_try.csv", "w")                                                  #Create submission file with writing access
#number_of_features = 20


#for i in range(number_of_features):                                             #Write stuff to the submission file
#    file.write(str(np.nan))
#    file.write('\n')
#file.close()
