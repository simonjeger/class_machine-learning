import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from scipy.special import expit
import pandas as pd
import inspect

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data():
    #read train features
    read_train_features = pd.read_csv('../data_2/train_features.csv', delimiter=',')
    read_train_features = read_train_features.replace('nan', np.NaN)
    read_train_features.fillna(read_train_features.mean(), inplace=True)        #Dealing with "nan" values --> replacing them with the mean value of the whole column
    data_train_features = read_train_features.to_numpy()

    #read train labels
    read_train_labels = pd.read_csv('../data_2/train_labels.csv', delimiter=',')
    data_train_labels = read_train_labels.to_numpy()

    #read test data
    read_test_features = pd.read_csv('../data_2/test_features.csv', delimiter=',')
    read_test_features = read_test_features.replace('nan', np.NaN)
    read_test_features.fillna(read_test_features.mean(), inplace=True)          #Dealing with "nan" values --> replacing them with the mean value of the whole column
    data_test_features = read_test_features.to_numpy()

    pid = []
    Y_LABELS = []
    X_FEATURES = []
    X_TEST = []

    for row in data_train_features:                                             #from Age (column C) to pH (column AK)
        X_FEATURES.append(list(row[2:]))


    for row in data_train_labels:
        pid.append(int(row[0]))
        Y_LABELS.append(list(row[1:11]))
    #copy each row of Y_LABELS 12 times to match the size of X_FEATURES
    n = 12
    Y_LABELS_extended = []
    for j in range(len(Y_LABELS)):
        for i in range(n):
            Y_LABELS_extended.append(Y_LABELS[j])

    for row in data_test_features:
        X_TEST.append(list(row[2:]))


    return [pid, Y_LABELS_extended, X_FEATURES, X_TEST]

def predict(X_FEATURES, w):
    return expit(np.dot(X_FEATURES, w))


###---------------MAIN----------------------------------------------------------
[pid, Y_LABELS, X_FEATURES, X_TEST] = read_in_data()                            #Read the data from features file

#cutting the dataset for debugging purposes:
n_cutoff = 50
pid_cut = pid[0:n_cutoff]
Y_LABELS_cut = np.dot(Y_LABELS[0:n_cutoff*12], 100)
print(Y_LABELS_cut)
X_FEATURES_cut = X_FEATURES[0:n_cutoff*12]
X_TEST_cut = X_TEST[0:n_cutoff*12]

##Subtask 1:
##Setting up a model with multiclass labels (Y_LABELS can be a matrix instead of just an array)
#model = OneVsRestClassifier(svm.SVC()).fit(X_FEATURES, Y_LABELS)

model = OneVsRestClassifier(svm.SVC(kernel='sigmoid'))
model.fit(X_FEATURES_cut, Y_LABELS_cut)

#get weights coefficients
#w = model.get_params('estimator__coef0')
#print(w)

#print(len(X_FEATURES_cut[0]))
X_TEST_cut = np.array(X_TEST_cut)
y_pred_cut = model.predict_proba(X_TEST_cut)
for i in range(0,n_cutoff):
    print(y_pred_cut[12*i])


#7.)    Predicting the values for the differents tests ordered


#8.)    Writing the predictions to the sample.csv file


##Subtask 2:
#9.)    Train a model using the label sepsis (0 for no sepsis, 1 otherwise)


#10.)   Predicting the occurance of sepsis with the standard function model.predict
        #example:     if(model.predict([[X_FEATURES[i](Age), ..., X_FEATURES[i](pH) ]]))==0:


#11.)   Writing the prediction for sepsis to the sample.csv file





#file = open("sample_try.csv", "w")                                                  #Create submission file with writing access
#number_of_features = 20


#for i in range(number_of_features):                                             #Write stuff to the submission file
#    file.write(str(np.nan))
#    file.write('\n')
#file.close()
