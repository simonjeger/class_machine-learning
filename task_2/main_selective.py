import numpy as np
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeCV
from sklearn import svm
import my_chabis

#Hexerei for importing parameters from yaml file
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()

with open(args.yaml_file, 'rt') as fh:
    yaml_parameters = yaml.safe_load(fh)

### Initialize all parameters from yaml file
patients = yaml_parameters['patients']                                          #number of patients to train the model
activation_ntest = yaml_parameters['activation_ntest']      #If True (1): takes the number of tests made into account
activation_gradient = yaml_parameters['activation_gradient']        #If True (1): takes the increase of decrease of vitals during the hospital stay into account
predicting_method = yaml_parameters['predicting_method']                        #values: 'sigmoid' or 'probability'
processing_method = yaml_parameters['processing_method']                        #values: 'deterministic' or 'random'
selection_method = yaml_parameters['selection_method']                          #values: 'selective' or 'non-selective' (for only the specific values or all values for training) or
magicfactor_testnumber = yaml_parameters['magicfactor_testnumber']
magicfactor_vitalgradient = yaml_parameters['magicfactor_vitalgradient']

def read_in_data():
    read_train_labels = pd.read_csv('../data_2/train_labels_5000.csv', delimiter=',')

    Y_LABELS_1 = read_train_labels.iloc[0:patients,1:11]
    Y_LABELS_2 = read_train_labels.iloc[0:patients,11]
    Y_LABELS_3 = read_train_labels.iloc[0:patients,12:16]

    train_features = pd.read_csv('../data_2/train_features_5000.csv', delimiter=',')
    train_features = train_features.replace('nan', np.NaN)

    test_features = pd.read_csv('../data_2/validation_features.csv', delimiter=',')
    test_features = test_features.replace('nan', np.NaN)

    return np.asarray(Y_LABELS_1), np.asarray(Y_LABELS_2), np.asarray(Y_LABELS_3), train_features, test_features

def count_values(data_set_train, data_set_test):
    X_count_train, X_count_test = [], []
    for i in range(0, patients):                                                #for training set
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        ntest = []
        for j in range(0, data_set_train.shape[1]):
            ntest.append(np.count_nonzero(~np.isnan(np.array(X_Patient)[:,j]), axis=0))
        X_count_train.append(ntest)
    for i in range(0, int(data_set_test.shape[0]/12)):                          #for test set
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        ntest = []
        for j in range(0, data_set_test.shape[1]):
            ntest.append(np.count_nonzero(~np.isnan(np.array(X_Patient)[:,j]), axis=0))
        X_count_test.append(ntest)
    return np.asarray(X_count_train), np.asarray(X_count_test)

def gradient(data_set_train, data_set_test):
    X_gradient_train, X_gradient_test = [], []
    ### For data_set_train
    for i in range(0, patients):
        gradient_vitals = []
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        X_Patient.index = np.arange(0, 12)                                      #set index from 0 to 11 in every X_Patient
        first_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.first_valid_index)))
        last_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.last_valid_index)))
        for j in range(0, data_set_train.shape[1]):
            gradient_vitals.append(np.nan_to_num(np.array(X_Patient)[int(last_elem[j]), j] - np.array(X_Patient)[int(first_elem[j]), j]))
        X_gradient_train.append(gradient_vitals)
    X_gradient_train = np.asarray(X_gradient_train)
    #max_gradient_train = np.amax(X_gradient_train, axis=0)
    #for i in range(0,len(X_gradient_train)):
    #    for j in range(0,len(X_gradient_train[0])):
    #        if max_gradient_train[j] != 0:
    #            X_gradient_train[i,j] /= max_gradient_train[j]
    ### For data_set_test
    for i in range(0, int(data_set_test.shape[0]/12)):
        gradient_vitals = []
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        X_Patient.index = np.arange(0, 12)                                      #set index from 0 to 11 in every X_Patient
        first_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.first_valid_index)))
        last_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.last_valid_index)))
        for j in range(0, data_set_test.shape[1]):
            gradient_vitals.append(np.nan_to_num(np.array(X_Patient)[int(last_elem[j]), j] - np.array(X_Patient)[int(first_elem[j]), j]))
        X_gradient_test.append(gradient_vitals)
    X_gradient_test = np.asarray(X_gradient_test)
    #max_gradient_test = np.amax(X_gradient_test, axis=0)
    #for i in range(0,len(X_gradient_test)):
    #    for j in range(0,len(X_gradient_test[0])):
    #        if max_gradient_test[j] != 0:
    #            X_gradient_test[i,j] /= max_gradient_test[j]

    return np.asarray(X_gradient_train), np.asarray(X_gradient_test)

def concatenate_random(data_set_train, data_set_test):
    mean_train = np.array(data_set_train.mean())
    std_train = np.array(data_set_train.std())
    mean_test = np.array(data_set_test.mean())
    std_test = np.array(data_set_test.std())
    X_TRAIN, X_TEST = [], []
    pid_test = []
    for i in range(0, patients):
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        X_append = np.random.normal(np.array(X_Patient.mean()),np.array(X_Patient.std()),int(data_set_train.shape[1]))
        for j in range(0,len(X_append)):                                        #if there are nan values after having taken mean --> replace them with random value of global mean and std
            if np.isnan(X_append[j]):
                X_append[j] = np.random.normal(mean_train[j], std_train[j], 1)
        X_TRAIN.append(list(X_append))
    for i in range(0, int(data_set_test.shape[0]/12)):
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        pid_test.append(np.array(data_set_test)[12*i,0])
        X_append = np.random.normal(np.array(X_Patient.mean()),np.array(X_Patient.std()),int(data_set_test.shape[1]))
        for j in range(0,len(X_append)):                                        #if there are nan values after having taken mean --> replace them with random value of global mean and std
            if np.isnan(X_append[j]):
                X_append[j] = np.random.normal(mean_test[j], std_test[j], 1)
        X_TEST.append(list(X_append))
    return np.asarray(X_TRAIN), np.asarray(X_TEST), np.asarray(pid_test)

def concatenate_deterministic(data_set_train, data_set_test):
    mean_train = np.array(data_set_train.mean())
    std_train = np.array(data_set_train.std())
    mean_test = np.array(data_set_test.mean())
    std_test = np.array(data_set_test.std())
    X_TRAIN, X_TEST = [], []
    pid_test = []
    for i in range(0, patients):
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        X_append = X_Patient.mean()
        for j in range(0, len(X_append)):
            if np.isnan(X_append[j]):
                X_append[j] = mean_train[j]
        X_TRAIN.append(list(X_append))
    for i in range(0, int(data_set_test.shape[0]/12)):
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        pid_test.append(np.array(data_set_test)[12*i,0])
        X_append = X_Patient.mean()
        for j in range(0, len(X_append)):
            if np.isnan(X_append[j]):
                X_append[j] = mean_test[j]
        X_TEST.append(list(X_append))
    return np.asarray(X_TRAIN), np.asarray(X_TEST), np.asarray(pid_test)

def selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test):
    X_TRAIN_1,X_TRAIN_2,X_TRAIN_3,X_TEST_1,X_TEST_2,X_TEST_3 = [],[],[],[],[],[]
    X_count_train_1,X_count_train_2,X_count_train_3,X_count_test_1,X_count_test_2,X_count_test_3 = [],[],[],[],[],[]
    X_gradient_train_1,X_gradient_train_2,X_gradient_train_3,X_gradient_test_1,X_gradient_test_2,X_gradient_test_3 = [],[],[],[],[],[]
    for row in X_TRAIN:
        selected_labels_1 = [row[10],row[12],row[17],row[27],row[33],row[6],row[34],row[20],row[29],row[3]]     #BaseExcess, Fibrinogen, AST, Alkalinephos, Bilirubin_total, Lactate, Troponinl, SaO2, Bilirubin_direct, EtCO2
        selected_labels_2 = [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[23], row[24], row[25], row[26], row[27], row[29], row[30], row[31], row[33], row[34], row[35], row[36]] #all without vitals
        selected_labels_3 = [row[9],row[22],row[28],row[32]]                                                    #RRate, ABPm, SpO2, Heartrate
        X_TRAIN_1.append(selected_labels_1)
        X_TRAIN_2.append(selected_labels_2)
        X_TRAIN_3.append(selected_labels_3)
    for row in X_TEST:
        selected_labels_1 = [row[10],row[12],row[17],row[27],row[33],row[6],row[34],row[20],row[29],row[3]]
        selected_labels_2 = [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[23], row[24], row[25], row[26], row[27], row[29], row[30], row[31], row[33], row[34], row[35], row[36]]
        selected_labels_3 = [row[9],row[22],row[28],row[32]]
        X_TEST_1.append(selected_labels_1)
        X_TEST_2.append(selected_labels_2)
        X_TEST_3.append(selected_labels_3)
    for row in X_count_train:
        selected_labels_1 = [row[10],row[12],row[17],row[27],row[33],row[6],row[34],row[20],row[29],row[3]]
        selected_labels_2 = [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[23], row[24], row[25], row[26], row[27], row[29], row[30], row[31], row[33], row[34], row[35], row[36]]
        selected_labels_3 = [row[9],row[22],row[28],row[32]]
        X_count_train_1.append(selected_labels_1)
        X_count_train_2.append(selected_labels_2)
        X_count_train_3.append(selected_labels_3)
    for row in X_count_test:
        selected_labels_1 = [row[10],row[12],row[17],row[27],row[33],row[6],row[34],row[20],row[29],row[3]]
        selected_labels_2 = [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[23], row[24], row[25], row[26], row[27], row[29], row[30], row[31], row[33], row[34], row[35], row[36]]
        selected_labels_3 = [row[9],row[22],row[28],row[32]]
        X_count_test_1.append(selected_labels_1)
        X_count_test_2.append(selected_labels_2)
        X_count_test_3.append(selected_labels_3)
    for row in X_gradient_train:
        selected_labels_1 = [row[10],row[12],row[17],row[27],row[33],row[6],row[34],row[20],row[29],row[3]]
        selected_labels_2 = [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[23], row[24], row[25], row[26], row[27], row[29], row[30], row[31], row[33], row[34], row[35], row[36]]
        selected_labels_3 = [row[9],row[22],row[28],row[32]]
        X_gradient_train_1.append(selected_labels_1)
        X_gradient_train_2.append(selected_labels_2)
        X_gradient_train_3.append(selected_labels_3)
    for row in X_gradient_test:
        selected_labels_1 = [row[10],row[12],row[17],row[27],row[33],row[6],row[34],row[20],row[29],row[3]]
        selected_labels_2 = [row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[23], row[24], row[25], row[26], row[27], row[29], row[30], row[31], row[33], row[34], row[35], row[36]]
        selected_labels_3 = [row[9],row[22],row[28],row[32]]
        X_gradient_test_1.append(selected_labels_1)
        X_gradient_test_2.append(selected_labels_2)
        X_gradient_test_3.append(selected_labels_3)
    return np.asarray(X_TRAIN_1), np.asarray(X_TRAIN_2), np.asarray(X_TRAIN_3), np.asarray(X_TEST_1), np.asarray(X_TEST_2), np.asarray(X_TEST_3), np.asarray(X_count_train_1), np.asarray(X_count_train_2), np.asarray(X_count_train_3), np.asarray(X_count_test_1), np.asarray(X_count_test_2), np.asarray(X_count_test_3), np.asarray(X_gradient_train_1), np.asarray(X_gradient_train_2), np.asarray(X_gradient_train_3), np.asarray(X_gradient_test_1), np.asarray(X_gradient_test_2), np.asarray(X_gradient_test_3)

def non_selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test):
    X_TRAIN_1 = X_TRAIN[:,2:]
    X_TRAIN_2 = X_TRAIN[:,2:]
    X_TRAIN_3 = X_TRAIN[:,2:]
    X_TEST_1 = X_TEST[:,2:]
    X_TEST_2 = X_TEST[:,2:]
    X_TEST_3 = X_TEST[:,2:]
    X_count_train_1 = X_count_train[:,2:]
    X_count_train_2 = X_count_train[:,2:]
    X_count_train_3 = X_count_train[:,2:]
    X_count_test_1 = X_count_test[:,2:]
    X_count_test_2 = X_count_test[:,2:]
    X_count_test_3 = X_count_test[:,2:]
    X_gradient_train_1 = X_gradient_train[:,2:]
    X_gradient_train_2 = X_gradient_train[:,2:]
    X_gradient_train_3 = X_gradient_train[:,2:]
    X_gradient_test_1 = X_gradient_test[:,2:]
    X_gradient_test_2 = X_gradient_test[:,2:]
    X_gradient_test_3 = X_gradient_test[:,2:]
    return np.asarray(X_TRAIN_1), np.asarray(X_TRAIN_2), np.asarray(X_TRAIN_3), np.asarray(X_TEST_1), np.asarray(X_TEST_2), np.asarray(X_TEST_3), np.asarray(X_count_train_1), np.asarray(X_count_train_2), np.asarray(X_count_train_3), np.asarray(X_count_test_1), np.asarray(X_count_test_2), np.asarray(X_count_test_3), np.asarray(X_gradient_train_1), np.asarray(X_gradient_train_2), np.asarray(X_gradient_train_3), np.asarray(X_gradient_test_1), np.asarray(X_gradient_test_2), np.asarray(X_gradient_test_3)

#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[Y_LABELS_1, Y_LABELS_2, Y_LABELS_3, train_features, test_features] = read_in_data()


### Determine a dataset that contains the number of tests made
if activation_ntest == True:
    print('Compute ntest matrix...')
    [X_count_train, X_count_test] = count_values(train_features, test_features)            #object array
else:
    X_count_train, X_count_test = [], []

### Create a dataset that contains the gradient of the tests made for each column and patient
if activation_gradient == True:
    print('Compute gradient matrix...')
    [X_gradient_train, X_gradient_test] = gradient(train_features, test_features)#object array
else:
    X_gradient_train, X_gradient_test = [], []

### Concatenate and put in mean values according to deterministic or random
print('Concatenate ' + processing_method + 'ly...')
if processing_method == 'deterministic':
    [X_TRAIN, X_TEST, pid_test] = concatenate_deterministic(train_features, test_features)#object array
elif processing_method == 'random':
    [X_TRAIN, X_TEST, pid_test] = concatenate_random(train_features, test_features)
else:
    print('Non-valid processing method. You are a Lauch')

### Writing pid of test patients directly to csv files
file_pid_test = pd.DataFrame(data=pid_test)
file_pid_test.to_csv('temporary_predictions/pid_test.csv')

### Model data preparing
print('Model data is being prepared...')
if selection_method == 'selective':
    [X_TRAIN_1, X_TRAIN_2, X_TRAIN_3, X_TEST_1, X_TEST_2, X_TEST_3, X_count_train_1, X_count_train_2, X_count_train_3, X_count_test_1, X_count_test_2, X_count_test_3, X_gradient_train_1, X_gradient_train_2, X_gradient_train_3, X_gradient_test_1, X_gradient_test_2, X_gradient_test_3] = selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test)
elif selection_method == 'non-selective':
    [X_TRAIN_1, X_TRAIN_2, X_TRAIN_3, X_TEST_1, X_TEST_2, X_TEST_3, X_count_train_1, X_count_train_2, X_count_train_3, X_count_test_1, X_count_test_2, X_count_test_3, X_gradient_train_1, X_gradient_train_2, X_gradient_train_3, X_gradient_test_1, X_gradient_test_2, X_gradient_test_3] = non_selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test)
else:
    print('Non-valid processing method. You are a Lauch')

### Model training
print('---Started model training---')
print('model_1_real')
model_1_real = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
model_1_real.fit(X_TRAIN_1, Y_LABELS_1)
if activation_ntest == True:
print('model_1_ntest')
model_1_ntest = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
model_1_ntest.fit(X_count_train_1, Y_LABELS_1)
print('model_1_gradient')
model_1_gradient = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
model_1_gradient.fit(X_gradient_train_1, Y_LABELS_1)
print('model_2_real')
model_2_real = svm.SVC(kernel='linear', probability=True)
model_2_real.fit(X_TRAIN_2, Y_LABELS_2)
print('model_2_ntest')
model_2_ntest = svm.SVC(kernel='linear', probability=True)
model_2_ntest.fit(X_count_train_2, Y_LABELS_2)
print('model_2_gradient')
model_2_gradient = svm.SVC(kernel='linear', probability=True)
model_2_gradient.fit(X_gradient_train_2, Y_LABELS_2)
print('model_3_real')
model_3_real = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)
model_3_real.fit(X_TRAIN_3, Y_LABELS_3)
print('model_3_ntest')
model_3_ntest = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)
model_3_ntest.fit(X_count_train_3, Y_LABELS_3)
print('model_3_gradient')
model_3_gradient = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)
model_3_gradient.fit(X_gradient_train_3, Y_LABELS_3)
print('---started predicting---')
if predicting_method == 'sigmoid':
    y_1_real = 1/(1 + np.exp(-model_1_real.decision_function(X_TEST_1)))
    file_y_1_real = pd.DataFrame(data=y_1_real)
    file_y_1_real.to_csv('temporary_predictions/y_1_real.csv')
    #y_1_ntest = 1/(1 + np.exp(-model_1_ntest.decision_function(X_count_test_1)))
    y_1_ntest = model_1_ntest.predict_proba(X_gradient_test_1)                  #alternative solution
    file_y_1_ntest = pd.DataFrame(data=y_1_ntest)
    file_y_1_ntest.to_csv('temporary_predictions/y_1_ntest.csv')
    #y_1_gradient = 1/(1 + np.exp(-model_1_gradient.decision_function(X_gradient_test_1)))
    y_1_gradient = model_1_gradient.predict_proba(X_gradient_test_1)            #alternative solution
    file_y_1_gradient = pd.DataFrame(data=y_1_gradient)
    file_y_1_gradient.to_csv('temporary_predictions/y_1_gradient.csv')
    y_2_real = 1/(1 + np.exp(-model_2_real.decision_function(X_TEST_2)))
    file_y_2_real = pd.DataFrame(data=y_2_real)
    file_y_2_real.to_csv('temporary_predictions/y_2_real.csv')
    y_2_ntest = model_2_ntest.predict_proba(X_gradient_test_2)[:,1]             #alternative solution
    #y_2_ntest = 1/(1 + np.exp(-model_2_ntest.decision_function(X_count_test_2)))
    file_y_2_ntest = pd.DataFrame(data=y_2_ntest)
    file_y_2_ntest.to_csv('temporary_predictions/y_2_ntest.csv')
    #y_2_gradient = 1/(1 + np.exp(-model_2_gradient.decision_function(X_gradient_test_2)))
    y_2_gradient = model_2_gradient.predict_proba(X_gradient_test_2)[:,1]       #alternative solution
    file_y_2_gradient = pd.DataFrame(data=y_2_gradient)
    file_y_2_gradient.to_csv('temporary_predictions/y_2_gradient.csv')
    y_3_real = model_3_real.predict(X_TEST_3)
    file_y_3_real = pd.DataFrame(data=y_3_real)
    file_y_3_real.to_csv('temporary_predictions/y_3_real.csv')
    y_3_ntest = model_3_ntest.predict(X_count_test_3)
    file_y_3_ntest = pd.DataFrame(data=y_3_ntest)
    file_y_3_ntest.to_csv('temporary_predictions/y_3_ntest.csv')
    y_3_gradient = model_3_gradient.predict(X_gradient_test_3)
    file_y_3_gradient = pd.DataFrame(data=y_3_gradient)
    file_y_3_gradient.to_csv('temporary_predictions/y_3_gradient.csv')
elif predicting_method == 'probability':
    y_1_real = model_1_real.predict_proba(X_TEST_1)
    file_y_1_real = pd.DataFrame(data=y_1_real)
    file_y_1_real.to_csv('temporary_predictions/y_1_real.csv')
    y_1_ntest = model_1_ntest.predict_proba(X_count_test_1)
    file_y_1_ntest = pd.DataFrame(data=y_1_ntest)
    file_y_1_ntest.to_csv('temporary_predictions/y_1_ntest.csv')
    y_1_gradient = model_1_gradient.predict_proba(X_gradient_test_1)
    file_y_1_gradient = pd.DataFrame(data=y_1_gradient)
    file_y_1_gradient.to_csv('temporary_predictions/y_1_gradient.csv')
    y_2_real = model_2_real.predict_proba(X_TEST_2)[:,1]
    file_y_2_real = pd.DataFrame(data=y_2_real)
    file_y_2_real.to_csv('temporary_predictions/y_2_real.csv')
    y_2_ntest = model_2_ntest.predict_proba(X_count_test_2)[:,1]
    file_y_2_ntest = pd.DataFrame(data=y_2_ntest)
    file_y_2_ntest.to_csv('temporary_predictions/y_2_ntest.csv')
    y_2_gradient = model_2_gradient.predict_proba(X_gradient_test)[:,1]
    file_y_2_gradient = pd.DataFrame(data=y_2_gradient)
    file_y_2_gradient.to_csv('temporary_predictions/y_2_gradient.csv')
    y_3_real = model_3_real.predict(X_TEST_3)
    file_y_3_real = pd.DataFrame(data=y_3_real)
    file_y_3_real.to_csv('temporary_predictions/y_3_real.csv')
    y_3_ntest = model_3_ntest.predict(X_count_test_3)
    file_y_3_ntest = pd.DataFrame(data=y_3_ntest)
    file_y_3_ntest.to_csv('temporary_predictions/y_3_ntest.csv')
    y_3_gradient = model_3_gradient.predict(X_gradient_test_3)
    file_y_3_gradient = pd.DataFrame(data=y_3_gradient)
    file_y_3_gradient.to_csv('temporary_predictions/y_3_gradient.csv')

print('---prediction files written---')

### Printing the whole stuff to files
#state_prediction = my_chabis.writing_prediction_files(pid_test, y_1_real, y_1_ntest, y_2_real, y_2_ntest, y_3_real)


### Moving steps towards ntest and gradient
#y_1 = np.add(y_1_real,magicfactor_testnumber*np.subtract(y_1_ntest,y_1_real))   #first step towards ntest
#y_2 = np.add(y_2_real,magicfactor_testnumber*np.subtract(y_2_ntest,y_2_real))
#y_3 = np.add(y_3_real,magicfactor_testnumber*np.subtract(y_3_ntest,y_3_real))
#y_1 = np.add(y_1, magicfactor_vitalgradient*np.subtract(y_1_gradient,y_1))      #second step towards gradient
#y_2 = np.add(y_2, magicfactor_vitalgradient*np.subtract(y_2_gradient,y_2))
#y_3 = np.add(y_3, magicfactor_vitalgradient*np.subtract(y_3_gradient,y_3))

### Writing to .zip and .csv files
#M_submission = np.c_[pid_test, y_1, y_2, y_3]
#M_submission_pd = pd.DataFrame(data=M_submission, columns=["pid","LABEL_BaseExcess","LABEL_Fibrinogen","LABEL_AST","LABEL_Alkalinephos","LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2","LABEL_Bilirubin_direct","LABEL_EtCO2","LABEL_Sepsis","LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"])

#M_Sub_panda.to_csv(r'sample_' + str(patients) + '_' + str(bool(activation_ntest)) + '_' + predicting_method + '_' + processing_method + '.csv', index=False, float_format='%.3f')
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('selective_' + str(patients) + '_' + str(bool(activation_ntest)) + '_' + predicting_method + '_' + processing_method + '.zip', index=False, float_format='%.3f', compression=compression_opts)
