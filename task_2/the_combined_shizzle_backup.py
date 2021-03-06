import numpy as np
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn import svm
import method_NN as NN

#Hexerei for importing parameters from yaml file
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()

with open(args.yaml_file, 'rt') as fh:
    yaml_parameters = yaml.safe_load(fh)

### Initialize all parameters from yaml file
patients = yaml_parameters['patients']                                          #number of patients to train the model
activation_ntest = yaml_parameters['activation_ntest']                          #If True (1): takes the number of tests made into account
activation_gradient = yaml_parameters['activation_gradient']                    #If True (1): takes the increase of decrease of vitals during the hospital stay into account
predicting_method = yaml_parameters['predicting_method']                        #values: 'sigmoid' or 'probability'
processing_method = yaml_parameters['processing_method']                        #values: 'deterministic' or 'random'
state_quick_calc = yaml_parameters['state_quick_calc']
activation_NN = yaml_parameters['activation_NN']

def read_in_data():
    train_labels = pd.read_csv('../data_2/train_labels.csv', delimiter=',', nrows=patients)

    Y_LABELS_1 = train_labels.iloc[:,1:11]
    Y_LABELS_2 = train_labels.iloc[:,11]
    Y_LABELS_3 = train_labels.iloc[:,12:16]

    train_features = pd.read_csv('../data_2/train_features.csv', delimiter=',', nrows=(12*patients))
    train_features = train_features.replace('nan', np.NaN)
    mean_global_train = train_features.mean()
    std_global_train = train_features.std()

    test_features = pd.read_csv('../data_2/validation_features.csv', delimiter=',')
    test_features = test_features.replace('nan', np.NaN)
    mean_global_test = test_features.mean()
    std_global_test = test_features.std()
    pid_test = test_features.iloc[::12,0]


    #Normalize the data
    #train_features = (train_features - mean_global_train)/std_global_train
    #print(test_features)
    #test_features = (test_features - mean_global_test)/std_global_test
    #print(test_features)

    return np.asarray(pid_test), np.asarray(Y_LABELS_1), np.asarray(Y_LABELS_2), np.asarray(Y_LABELS_3), train_features, test_features, np.asarray(mean_global_train), np.asarray(std_global_train), np.asarray(mean_global_test), np.asarray(std_global_test)

def count_values(data_set_train, data_set_test):
    X_count_train, X_count_test = [], []
    ntest = []
    for i in range(0, patients):                                                #for training set
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        if activation_ntest == 1:                                               #leave out pid, Time, Age and the fantastic four vitals
            ntest = np.count_nonzero(~np.isnan(np.array(X_Patient)[:,:]), axis=0)
            ntest = ntest.astype(bool)*1
        elif activation_ntest == 2:
            ntest = np.count_nonzero(~np.isnan(np.array(X_Patient)[:,:]))
        X_count_train.append(ntest)
    for i in range(0, int(data_set_test.shape[0]/12)):                          #for test set
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        if activation_ntest == 1:                                               #leave out pid, Time, Age and the fantastic four vitals
            ntest = np.count_nonzero(~np.isnan(np.array(X_Patient)[:,:]), axis=0)
            ntest = ntest.astype(bool)*1
        elif activation_ntest == 2:
            ntest = np.count_nonzero(~np.isnan(np.array(X_Patient)[:,:]))
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
        for j in range(3, data_set_train.shape[1]):                             #leave out pid, Time and Age
            gradient_vitals.append(np.nan_to_num(np.array(X_Patient)[int(last_elem[j]), j] - np.array(X_Patient)[int(first_elem[j]), j]))
        X_gradient_train.append(gradient_vitals)
    X_gradient_train = np.asarray(X_gradient_train)

    for i in range(0, int(data_set_test.shape[0]/12)):
        gradient_vitals = []
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        X_Patient.index = np.arange(0, 12)                                      #set index from 0 to 11 in every X_Patient
        first_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.first_valid_index)))
        last_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.last_valid_index)))
        for j in range(3, data_set_test.shape[1]):
            gradient_vitals.append(np.nan_to_num(np.array(X_Patient)[int(last_elem[j]), j] - np.array(X_Patient)[int(first_elem[j]), j]))
        X_gradient_test.append(gradient_vitals)
    X_gradient_test = np.asarray(X_gradient_test)

    return np.asarray(X_gradient_train), np.asarray(X_gradient_test)

def concatenate_deterministicly(data_set_train, data_set_test, mean_global_train, std_global_train, mean_global_test, std_global_test):
    X_TRAIN, X_TEST = [], []
    #for data_set_train
    for i in range(0, patients):
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        X_append = X_Patient.mean()
        for j in range(0, len(X_append)):
            if np.isnan(X_append[j]):
                X_append[j] = mean_global_train[j]
        X_TRAIN.append(list(X_append))
    #for data_set_test
    for i in range(0, int(data_set_test.shape[0]/12)):
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        X_append = X_Patient.mean()
        for j in range(0, len(X_append)):
            if np.isnan(X_append[j]):
                X_append[j] = mean_global_test[j]
        X_TEST.append(list(X_append))
    return np.asarray(X_TRAIN), np.asarray(X_TEST)

def concatenate_randomly(data_set_train, data_set_test, mean_global_train, std_global_train, mean_global_test, std_global_test):
    X_TRAIN, X_TEST = [], []
    #for data_set_train
    for i in range(0, patients):
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        X_append = np.random.normal(np.array(X_Patient.mean()),np.array(X_Patient.std()),int(data_set_train.shape[1]))
        for j in range(0,len(X_append)):                                        #if there are nan values after having taken mean --> replace them with random value of global mean and std
            if np.isnan(X_append[j]):
                X_append[j] = np.random.normal(mean_global_train[j], std_global_train[j], 1)
        X_TRAIN.append(list(X_append))
    #for data_set_test
    for i in range(0, int(data_set_test.shape[0]/12)):
        X_Patient = data_set_test.iloc[(12*i):(12*(i+1)),:]
        X_append = np.random.normal(np.array(X_Patient.mean()),np.array(X_Patient.std()),int(data_set_test.shape[1]))
        for j in range(0,len(X_append)):                                        #if there are nan values after having taken mean --> replace them with random value of global mean and std
            if np.isnan(X_append[j]):
                X_append[j] = np.random.normal(mean_global_test[j], std_global_test[j], 1)
        X_TEST.append(list(X_append))
    return np.asarray(X_TRAIN), np.asarray(X_TEST)

def selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test):
    X_TRAIN_1,X_TRAIN_2,X_TRAIN_3,X_TEST_1,X_TEST_2,X_TEST_3 = [],[],[],[],[],[]
    #for train data
    X_TRAIN_1 = np.c_[X_TRAIN[:,10], X_TRAIN[:,12], X_TRAIN[:,17], X_TRAIN[:,27], X_TRAIN[:,33], X_TRAIN[:,6], X_TRAIN[:,34], X_TRAIN[:,20], X_TRAIN[:,29], X_TRAIN[:,3]]
    X_TRAIN_2 = X_TRAIN[:,2:36]
    X_TRAIN_3 = np.c_[X_TRAIN[:,9], X_TRAIN[:,22], X_TRAIN[:,28], X_TRAIN[:,32]]    #RRate, ABPm, SpO2, Heartrate
    #if activation_ntest != 0:
    #    X_TRAIN_1 = np.c_[X_TRAIN_1, X_count_train]
    #    X_TRAIN_2 = np.c_[X_TRAIN_2, X_count_train]
    if activation_ntest == 1:
        X_TRAIN_1 = np.c_[X_TRAIN_1, X_count_train[:,10], X_count_train[:,12], X_count_train[:,17], X_count_train[:,27], X_count_train[:,33], X_count_train[:,6], X_count_train[:,34], X_count_train[:,20], X_count_train[:,29], X_count_train[:,3]]
        #X_TRAIN_3 = np.c_[X_TRAIN_3, X_count_train[:,6], X_count_train[:,19], X_count_train[:,25], X_count_train[:,29]]
    #if activation_ntest == 2:
    #    X_TRAIN_3 = np.c_[X_TRAIN_3, X_count_train]
    if activation_gradient == 1:
        #X_TRAIN_1 = np.c_[X_TRAIN_1, X_gradient_train]
        #X_TRAIN_2 = np.c_[X_TRAIN_2, X_gradient_train]
        X_TRAIN_3 = np.c_[X_TRAIN_3, X_gradient_train[:,6], X_gradient_train[:,19], X_gradient_train[:,25], X_gradient_train[:,29]]

    #for test data
    X_TEST_1 = np.c_[X_TEST[:,10], X_TEST[:,12], X_TEST[:,17], X_TEST[:,27], X_TEST[:,33], X_TEST[:,6], X_TEST[:,34], X_TEST[:,20], X_TEST[:,29], X_TEST[:,3]]
    X_TEST_2 = X_TEST[:,2:36]
    X_TEST_3 = np.c_[X_TEST[:,9], X_TEST[:,22], X_TEST[:,28], X_TEST[:,32]]     #RRate, ABPm, SpO2, Heartrate
    #if activation_ntest != 0:
    #    X_TEST_1 = np.c_[X_TEST_1, X_count_test]
    #    X_TEST_2 = np.c_[X_TEST_2, X_count_test]
    if activation_ntest == 1:
        X_TEST_1 = np.c_[X_TEST_1, X_count_test[:,10], X_count_test[:,12], X_count_test[:,17], X_count_test[:,27], X_count_test[:,33], X_count_test[:,6], X_count_test[:,34], X_count_test[:,20], X_count_test[:,29], X_count_test[:,3]]
    #if activation_ntest == 2:
    #    X_TEST_3 = np.c_[X_TEST_3, X_count_test]
    if activation_gradient == 1:
        #X_TEST_1 = np.c_[X_TEST_1, X_gradient_test]
        #X_TEST_2 = np.c_[X_TEST_2, X_gradient_test]
        X_TEST_3 = np.c_[X_TEST_3, X_gradient_test[:,6], X_gradient_test[:,19], X_gradient_test[:,25], X_gradient_test[:,29]]

    return np.asarray(X_TRAIN_1), np.asarray(X_TRAIN_2), np.asarray(X_TRAIN_3), np.asarray(X_TEST_1), np.asarray(X_TEST_2), np.asarray(X_TEST_3)


#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[pid_test, Y_LABELS_1, Y_LABELS_2, Y_LABELS_3, train_features, test_features, mean_global_train, std_global_train, mean_global_test, std_global_test] = read_in_data()

### Determine a dataset that contains the number of tests made
if activation_ntest != 0:
    print('Compute ntest matrix...')
    [X_count_train, X_count_test] = count_values(train_features, test_features)
else:
    X_count_train, X_count_test = np.asarray([]), np.asarray([])

### Create a dataset that contains the gradient of the tests made for each column and patient
if activation_gradient == 1:
    print('Compute gradient matrix...')
    [X_gradient_train, X_gradient_test] = gradient(train_features, test_features)
else:
    X_gradient_train, X_gradient_test = np.asarray([]), np.asarray([])

### Concatenate and put in mean values according to deterministic or random
if state_quick_calc == False:
    print('Concatenate ' + processing_method + 'ly...')
    if processing_method == 'deterministic':
        [X_TRAIN, X_TEST] = concatenate_deterministicly(train_features, test_features, mean_global_train, std_global_train, mean_global_test, std_global_test)#object array
    elif processing_method == 'random':
        [X_TRAIN, X_TEST] = concatenate_randomly(train_features, test_features, mean_global_train, std_global_train, mean_global_test, std_global_test)
    train_pandas = pd.DataFrame(X_TRAIN)
    train_pandas.to_csv('../data_2/patient_data_train_prep.csv', header=False, index=False) #store data to save time
    test_pandas = pd.DataFrame(X_TEST)
    test_pandas.to_csv('../data_2/patient_data_test_prep.csv', header=False, index=False) #store data to save time
else:
    X_TRAIN = np.asarray(pd.read_csv("../data_2/patient_data_train_prep.csv", header=None))             #Read in previously generated data
    X_TEST = np.asarray(pd.read_csv("../data_2/patient_data_test_prep.csv", header=None))
    print(pd.read_csv("../data_2/patient_data_test_prep.csv", header=None))

### Model data preparing
print('Model data is being prepared...')
[X_TRAIN_1, X_TRAIN_2, X_TRAIN_3, X_TEST_1, X_TEST_2, X_TEST_3] = selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test)

print('---Started model training---')
print('model_1:')
if activation_NN == True:
    y_1 = NN.neural_network(X_TRAIN_1, Y_LABELS_1, X_TEST_1)
else:
    model_1 = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    model_1.fit(X_TRAIN_1, Y_LABELS_1)
    if predicting_method == 'sigmoid':
        y_1 = 1/(1 + np.exp(-model_1.decision_function(X_TEST_1)))
    elif predicting_method == 'probability':
        y_1 = model_1.predict_proba(X_TEST_1)

print('model_2:')
model_2 = svm.SVC(kernel='linear', probability=True)
model_2.fit(X_TRAIN_2, Y_LABELS_2)
if predicting_method == 'sigmoid':
    y_2 = 1/(1 + np.exp(-model_2.decision_function(X_TEST_2)))
elif predicting_method == 'probability':
    y_2 = model_2.predict_proba(X_TEST_2)[:,1]

print('model_3:')
model_3 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)
model_3.fit(X_TRAIN_3, Y_LABELS_3)
y_3 = model_3.predict(X_TEST_3)

### Writing to .zip and .csv files
M_submission = np.c_[pid_test, y_1, y_2, y_3]
M_submission_pd = pd.DataFrame(data=M_submission, columns=["pid","LABEL_BaseExcess","LABEL_Fibrinogen","LABEL_AST","LABEL_Alkalinephos","LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2","LABEL_Bilirubin_direct","LABEL_EtCO2","LABEL_Sepsis","LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"])

M_submission_pd.to_csv(r'sample_' + str(patients) + '_' + str(activation_ntest) + '_' + str(activation_gradient) + '_' + predicting_method + '.csv', index=False, float_format='%.3f')
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample_' + str(patients) + '_' + str(activation_ntest) + '_' + predicting_method + '_' + processing_method + '.zip', index=False, float_format='%.3f', compression=compression_opts)
