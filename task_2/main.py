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
    mean_train = read_train_features.mean()
    std_train = read_train_features.std()
    number_of_patients = int(read_train_features.shape[0]/12)
    [X_TRAIN, pid_train] = pre_processing(number_of_patients, mean_train, std_train, read_train_features)

    #read train labels (pid and Y_LABELS)
    read_train_labels = pd.read_csv('../data_2/train_labels.csv', delimiter=',', nrows=10)
    data_train_labels = read_train_labels.to_numpy()

    Y_LABELS = []
    for row in data_train_labels:
        Y_LABELS.append(list(row[1:11]))

    #read test data (X_TEST)
    read_test_features = pd.read_csv('../data_2/test_features.csv', delimiter=',', nrows=120)
    read_test_features = read_test_features.replace('nan', np.NaN)
    mean_test = read_test_features.mean()
    std_test = read_test_features.std()
    [X_TEST, pid_test] = pre_processing(number_of_patients, mean_test, std_test, read_test_features)

    return [pid_train, pid_test, Y_LABELS, X_TRAIN, X_TEST]

def pre_processing(number_of_patients, mean, std, data_set):
    X_CUT = []
    pid = []
    for i in range(0,number_of_patients):                                       #going from 1 through all patients
        X_Patient = data_set.iloc[(12*i):(12*(i+1)),:]                          #extract only the relevant data for that specific patient
        X_append = X_Patient.mean()
        X_append = X_append.to_numpy()                                          #Has still all the columns in it
        pid.append(X_append[0])
        X_append = X_append[2:]                                                 #cut away the pid and the time (start from Age)

        for j in range(0,len(X_append)):
            if np.isnan(X_append[j]):
                X_append[j] = np.random.normal(mean, std, 1)
        X_CUT.append(list(X_append))

    return [X_CUT, pid]


###---------------MAIN----------------------------------------------------------
[pid_train, pid_test, Y_LABELS, X_TRAIN, X_TEST] = read_in_data()            #Read the data from features file
X_TEST = np.nan_to_num(X_TEST)#random stuff:)


##Subtask 1: Setting up a model with multiclass labels
model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
model.fit(X_TRAIN, Y_LABELS)

#Prediction
y_pred = model.predict_proba(X_TEST)
print(y_pred)



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
