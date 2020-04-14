import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeCV
from sklearn import svm
import pandas as pd

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data(patients, activation_number_of_tests):
    #read train labels (Y_LABELS)
    read_train_labels = pd.read_csv('../data_2/train_labels.csv', delimiter=',')
    data_train_labels = np.array(read_train_labels)
    Y_LABELS_1 = []
    Y_LABELS_2 = []
    Y_LABELS_3 = []
    for row in data_train_labels:
        Y_LABELS_1.append(list(row[1:11]))
        Y_LABELS_2.append(float(row[11]))
        Y_LABELS_3.append(list(row[12:16]))

    #read train features (X_TRAIN)
    read_train_features = pd.read_csv('../data_2/train_features.csv', delimiter=',')
    read_train_features = read_train_features.replace('nan', np.NaN)            #MAGIE
    mean_train = np.array(read_train_features.mean())                           #get mean and std over all patients for filling in nan's
    std_train = np.array(read_train_features.std())
    number_train_patients = int(read_train_features.shape[0]/12)
    [X_TRAIN, pid_train] = pre_processing(number_train_patients, mean_train, std_train, read_train_features, activation_number_of_tests)        #pull X_TRAIN together for only one row per patient

    [X_TRAIN, Y_LABELS_1, Y_LABELS_2, Y_LABELS_3] = shuffle(X_TRAIN, Y_LABELS_1, Y_LABELS_2, Y_LABELS_3, n_samples=patients)      #shuffling and taking only the desired number of patients for training

    #read test data (X_TEST)
    read_test_features = pd.read_csv('../data_2/test_features.csv', delimiter=',')
    read_test_features = read_test_features.replace('nan', np.NaN)
    mean_test = np.array(read_test_features.mean())
    std_test = np.array(read_test_features.std())
    number_test_patients = int(read_test_features.shape[0]/12)
    [X_TEST, pid_test] = pre_processing(number_test_patients, mean_test, std_test, read_test_features, activation_number_of_tests)

    return [pid_train, pid_test, Y_LABELS_1, Y_LABELS_2, Y_LABELS_3, X_TRAIN, X_TEST]

def pre_processing(number_of_patients, mean, std, data_set, activation_number_of_tests):
    X_CUT = []
    pid = []

    for i in range(0,number_of_patients):                                       #going from 1 through all patients
        X_Patient = data_set.iloc[(12*i):(12*(i+1)),:]                          #extract only the relevant data for that specific patient
        X_append = X_Patient.mean()
        X_append = np.array(X_append)                                           #Has still all the columns in it
        pid.append(X_append[0])
        X_append = X_append[2:]                                                 #cut away the pid and the time (start from Age)

        for j in range(0,len(X_append)):
            if np.isnan(X_append[j]):
                np.random.seed(1)
                X_append[j] = np.random.normal(mean[j+2], std[j+2], 1)

        if activation_number_of_tests == True:
            number_of_tests_made = 0
            number_of_tests_made = np.count_nonzero(~np.isnan(np.array(X_Patient)[0:12,3:len(X_append)+2]))     #number of non-NaN values in all tests over all hours
            X_append = np.append(X_append, number_of_tests_made)

        X_CUT.append(list(X_append))

    return [X_CUT, pid]


###---------------MAIN----------------------------------------------------------
patients = 50                           #number of patients to train the model
activation_number_of_tests = True       #if true it takes the number of tests made into account

[pid_train, pid_test, Y_LABELS_1, Y_LABELS_2, Y_LABELS_3, X_TRAIN, X_TEST] = read_in_data(patients, activation_number_of_tests)            #Read the data from features file
X_TEST = np.nan_to_num(X_TEST)          #random stuff:)

##Subtask 1: Setting up a model with multiclass labels
model_1 = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
model_1.fit(np.array(X_TRAIN), np.array(Y_LABELS_1))
y_pred_1 = model_1.predict_proba(X_TEST)                                        #Prediction


##Subtask 2: Setting up a model for sepsis
model_2 = svm.SVC(kernel='linear', probability=True)
model_2.fit(np.array(X_TRAIN), np.array(Y_LABELS_2))
y_pred_2 = model_2.predict_proba(X_TEST)[:,1]                                   #Prediction

##Subtask 3: Setting up a model for mean of vital signs
model_3 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)                        #None for Leave-One-Out cross-validation
model_3.fit(np.array(X_TRAIN), np.array(Y_LABELS_3))
y_pred_3 = model_3.predict(X_TEST)                                              #Prediction

#Creating submission matrix and writing to zip
M_Sub = np.c_[pid_test, y_pred_1, y_pred_2, y_pred_3]
M_Sub_panda = pd.DataFrame(data=M_Sub, columns=["pid","LABEL_BaseExcess","LABEL_Fibrinogen","LABEL_AST","LABEL_Alkalinephos","LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2","LABEL_Bilirubin_direct","LABEL_EtCO2","LABEL_Sepsis","LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"])

M_Sub_panda.to_csv(r'sample_' + str(patients) + '.csv', index = False)
compression_opts = dict(method='zip', archive_name='sample.csv')
M_Sub_panda.to_csv('sample_' + str(patients) + '.zip', index=False, float_format='%.3f', compression=compression_opts)
