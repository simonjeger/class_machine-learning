import numpy as np
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
import pandas as pd
from sklearn.utils import shuffle
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn import svm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import normalize, to_categorical

import method_NN as NN

from scipy import stats

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
activation_NN_1 = yaml_parameters['activation_NN_1']
activation_NN_2 = yaml_parameters['activation_NN_2']
activation_NN_3 = yaml_parameters['activation_NN_3']

### FOR CHOOSING THE RIGHT HYPERPARAMS
hypertraining_iterations = 10
activation_hypertuning = True
if activation_hypertuning==True:
    hyperarr_epochs = [10]                                                      #old sets: [1, 5, 10, 20, 30]
    hyperarr_batchsize = [32, 64, 128]                                          #old sets: [20, 32, 32, 50, 75]
    hyperarr_n_layers = [2, 3]                                                  #old sets: [1, 2, 3, 4]
    hyperarr_start_density = [100, 200, 300]                                    #old sets: [100, 300, 500]
    hyperarr_dropout = [0.1, 0.2, 0.3]                                          #old sets: [0.1, 0.3, 0.5]
else:
    hyperarr_epochs = [5]
    hyperarr_batchsize = [32]
    hyperarr_n_layers = [3]
    hyperarr_start_density = [200]
    hyperarr_dropout = [0.2]

def read_in_data():
    train_labels = pd.read_csv('../data_2/train_labels.csv', delimiter=',', nrows=patients)

    Y_LABELS_1 = train_labels.iloc[:,1:11]
    Y_LABELS_2 = train_labels.iloc[:,11]
    Y_LABELS_3 = train_labels.iloc[:,12:16]

    train_features = pd.read_csv('../data_2/train_features.csv', delimiter=',', nrows=(12*patients))
    train_features = train_features.replace('nan', np.NaN)
    mean_global_train = train_features.mean()
    std_global_train = train_features.std()

    test_features = pd.read_csv('../data_2/test_features.csv', delimiter=',')
    test_features = test_features.replace('nan', np.NaN)
    mean_global_test = test_features.mean()
    std_global_test = test_features.std()
    pid_test = test_features.iloc[::12,0]

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

def slope(data_set, type):
    X_gradient_train, X_gradient_test = [], []
    result = []

    if type == 'train':
        N = patients
    if type == 'test':
        N = int(data_set.shape[0]/12)

    for i in range(0, N):
        gradient_vitals = []
        X_Patient = data_set.iloc[(12*i):(12*(i+1)),:]
        X_Patient.index = np.arange(0, 12)
        X_Patient = np.array(X_Patient)                                      #set index from 0 to 11 in every X_Patient
        X_Patient_result = [0] * (data_set.shape[1]-3)
        for j in range(3, data_set.shape[1]):                             #leave out pid, Time and Age
            X_vector = X_Patient[:,j]
            is_not_nan = ~np.isnan(X_vector)
            Y_vector = []
            for k in range(0,len(X_vector)):
                if is_not_nan[k]:
                    Y_vector.append(k)
            X_vector = X_vector[~np.isnan(X_vector)]
            if X_vector.any():
                if len(X_vector) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X_vector,Y_vector)
                    if np.isnan(slope):
                        slope = 0
                else:
                    slope = 0
            else:
                slope = 0
            X_Patient_result[j-3] = slope
        result.append(X_Patient_result)
    return result

def gradient(data_set_train, data_set_test):
    X_gradient_train, X_gradient_test = [], []
    ### For data_set_train
    for i in range(0, patients):
        gradient_vitals = []
        X_Patient = data_set_train.iloc[(12*i):(12*(i+1)),:]
        X_Patient.index = np.arange(0, 12)                                      #set index from 0 to 11 in every X_Patient
        first_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.first_valid_index)))
        last_elem = np.nan_to_num(np.asarray(X_Patient.apply(pd.Series.last__index)))
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
    X_TRAIN_2 = X_TRAIN[:,2:]
    #X_TRAIN_2 = np.c_[X_TRAIN[:,2], X_TRAIN[:,3], X_TRAIN[:,6], X_TRAIN[:,7], X_TRAIN[:,11], X_TRAIN[:,13], X_TRAIN[:,21], X_TRAIN[:,32], X_TRAIN[:,36]]
    X_TRAIN_3 = np.c_[X_TRAIN[:,9], X_TRAIN[:,22], X_TRAIN[:,28], X_TRAIN[:,32]]    #RRate, ABPm, SpO2, Heartrate
    if activation_ntest == 1:
        X_TRAIN_1 = np.c_[X_TRAIN_1, X_count_train[:,10], X_count_train[:,12], X_count_train[:,17], X_count_train[:,27], X_count_train[:,33], X_count_train[:,6], X_count_train[:,34], X_count_train[:,20], X_count_train[:,29], X_count_train[:,3]]
        #X_TRAIN_2 = np.c_[X_TRAIN_2, X_count_train[:,2:]]
        #X_TRAIN_2 = np.c_[X_TRAIN_2, X_count_train[:,2], X_count_train[:,6], X_count_train[:,7], X_count_train[:,11], X_count_train[:,13], X_count_train[:,21], X_count_train[:,32], X_count_train[:,36]]
    if activation_gradient != 0:
        X_TRAIN_1 = np.c_[X_TRAIN_1, X_gradient_train]
        X_TRAIN_2 = np.c_[X_TRAIN_2, X_gradient_train]
        X_TRAIN_3 = np.c_[X_TRAIN_3, X_gradient_train]
    #for test data
    X_TEST_1 = np.c_[X_TEST[:,10], X_TEST[:,12], X_TEST[:,17], X_TEST[:,27], X_TEST[:,33], X_TEST[:,6], X_TEST[:,34], X_TEST[:,20], X_TEST[:,29], X_TEST[:,3]]
    X_TEST_2 = X_TEST[:,2:]
    #X_TEST_2 = np.c_[X_TEST[:,2], X_TEST[:,3], X_TEST[:,6], X_TEST[:,7], X_TEST[:,11], X_TEST[:,13], X_TEST[:,21], X_TEST[:,32], X_TEST[:,36]]
    X_TEST_3 = np.c_[X_TEST[:,9], X_TEST[:,22], X_TEST[:,28], X_TEST[:,32]]     #RRate, ABPm, SpO2, Heartrate
    if activation_ntest == 1:
        X_TEST_1 = np.c_[X_TEST_1, X_count_test[:,10], X_count_test[:,12], X_count_test[:,17], X_count_test[:,27], X_count_test[:,33], X_count_test[:,6], X_count_test[:,34], X_count_test[:,20], X_count_test[:,29], X_count_test[:,3]]
        #X_TEST_2 = np.c_[X_TEST_2, X_count_test[:,2:]]
        #X_TEST_2 = np.c_[X_TEST_2, X_count_test[:,2], X_count_test[:,6], X_count_test[:,7], X_count_test[:,11], X_count_test[:,13], X_count_test[:,21], X_count_test[:,32], X_count_test[:,36]]
    if activation_gradient != 0:
        X_TEST_1 = np.c_[X_TEST_1, X_gradient_test]
        X_TEST_2 = np.c_[X_TEST_2, X_gradient_test]
        X_TEST_3 = np.c_[X_TEST_3, X_gradient_test]

    return np.asarray(X_TRAIN_1), np.asarray(X_TRAIN_2), np.asarray(X_TRAIN_3), np.asarray(X_TEST_1), np.asarray(X_TEST_2), np.asarray(X_TEST_3)

def evaluate(task, Y_LABELS_VALIDATION, y_validation):
    if task == 1:
        return metrics.roc_auc_score(Y_LABELS_VALIDATION, y_validation)
    if task == 2:
        return metrics.roc_auc_score(Y_LABELS_VALIDATION, y_validation)
    if task == 3:
        return 0.5 + 0.5 * np.maximum(0, metrics.r2_score(Y_LABELS_VALIDATION, y_validation))

#def loss_func_2(y_actual, y_predicted):
#    return metrics.roc_auc_score(np.asarray(y_actual), np.asarray(y_predicted))

class Val_2(Callback):
    def __init__(self, X, y):
        self.X_val = X
        self.y_val = y
        self.best_val_2 = 0

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.round(self.model.predict(self.X_val).reshape(-1))
        print(val_predict)
        accuracy_2 = metrics.roc_auc_score(self.y_val, val_predict)
        if accuracy_2 > self.best_val_2:            #if the value improves from one epoch to another, but still with the same set of hyper parameters
            self.best_val_2 = accuracy_2
            print('value improved')
        else:
            print('value kept the same')

        if accuracy_2 > score_2[0]:                 #if the value is better than all we have found so far with any settings of hyper parameters
            score_2[0] = accuracy_2
            best_variables = [hyperparam_epochs, hyperparam_batchsize, hyperparam_n_layers, hyperparam_start_density, hyperparam_dropout]
            print('new best score found:', best_variables)

#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[pid_test, Y_LABELS_1, Y_LABELS_2, Y_LABELS_3, train_features, test_features, mean_global_train, std_global_train, mean_global_test, std_global_test] = read_in_data()

### Determine a dataset that contains the number of tests made
if activation_ntest == 1:
    print('Compute ntest matrix...')
    [X_count_train, X_count_test] = count_values(train_features, test_features)
else:
    X_count_train, X_count_test = np.asarray([]), np.asarray([])

### Create a dataset that contains the gradient of the tests made for each column and patient
if activation_gradient == 1:
    print('Compute gradient matrix...')
    [X_gradient_train, X_gradient_test] = gradient(train_features, test_features)
elif activation_gradient == 2:
    print('Compute slope matrix...')
    X_gradient_train = slope(train_features, 'train')
    X_gradient_test = slope(test_features, 'test')
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

### Model data preparing
print('Model data is being prepared...')
[X_TRAIN_1, X_TRAIN_2, X_TRAIN_3, X_TEST_1, X_TEST_2, X_TEST_3] = selective_training(X_TRAIN, X_TEST, X_count_train, X_count_test, X_gradient_train, X_gradient_test)
X_TRAIN_1, X_TRAIN_1_VALIDATION, Y_LABELS_1, Y_LABELS_1_VALIDATION = train_test_split(X_TRAIN_1,Y_LABELS_1, test_size = 0.2)
X_TRAIN_2, X_TRAIN_2_VALIDATION, Y_LABELS_2, Y_LABELS_2_VALIDATION = train_test_split(X_TRAIN_2,Y_LABELS_2, test_size = 0.2)
X_TRAIN_3, X_TRAIN_3_VALIDATION, Y_LABELS_3, Y_LABELS_3_VALIDATION = train_test_split(X_TRAIN_3,Y_LABELS_3, test_size = 0.2)

print('---Started model training---')
print('model_1:')
if activation_NN_1 == True:
    tuning_params = []
    for n in range(1):
        ### SETTING RANDOM HYPERPARAMETERS
        hyperparam_epochs = hyperarr_epochs[np.random.randint(0,len(hyperarr_epochs))]
        hyperparam_batchsize = hyperarr_batchsize[np.random.randint(0,len(hyperarr_batchsize))]
        hyperparam_n_layers = hyperarr_n_layers[np.random.randint(0,len(hyperarr_n_layers))]
        hyperparam_start_density = hyperarr_start_density[np.random.randint(0,len(hyperarr_start_density))]
        hyperparam_dropout = hyperarr_dropout[np.random.randint(0,len(hyperarr_dropout))]

        ### GENERATING NN
        model_1 = tf.keras.models.Sequential()
        model_1.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN_1[0]),)))
        for n_layers in range(hyperparam_n_layers):
            model_1.add(tf.keras.layers.Dense(int(hyperparam_start_density/np.power(2,n_layers)), activation='sigmoid'))
            model_1.add(tf.keras.layers.Dropout(hyperparam_dropout))
            model_1.add(tf.keras.layers.BatchNormalization())
        model_1.add(tf.keras.layers.Dense(len(Y_LABELS_1[0]), activation='sigmoid'))
        model_1.summary()

        model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model_1.fit(X_TRAIN_1, Y_LABELS_1, batch_size=hyperparam_batchsize, epochs=hyperparam_epochs, verbose=1)
        y_validation = model_1.predict(X_TRAIN_1_VALIDATION)
        y_pred_test_1 = model_1.predict(X_TEST_1)
        del model_1

        accuracy = evaluate(1, Y_LABELS_1_VALIDATION, y_validation)
        tuning_params.append([hyperparam_epochs, hyperparam_batchsize, hyperparam_n_layers, hyperparam_start_density, hyperparam_dropout, accuracy])
        print(tuning_params)

        ### WRITING SOLUTION TO SUBMISSION .CSV-FILE
        M_submission_pd = pd.DataFrame(data=y_pred_test_1)
        M_submission_pd.to_csv(r'submission_subtask_1/submission_' + str(n) + '.csv', index=False, header=False)

    ### WRITING PARAMETERS TO FILE 'hyperparam_settings.csv'
    M_hyperparam_settings = pd.DataFrame(data=tuning_params, columns=["epochs", "batch_size", "n_layers_NN", "start_density", "dropout", "accuracy"])
    print(M_hyperparam_settings)
    M_hyperparam_settings.to_csv(r'hyperparam_settings_1.csv')

else:
    model_1 = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
    model_1.fit(X_TRAIN_1, Y_LABELS_1)
    if predicting_method == 'sigmoid':
        y_1 = 1/(1 + np.exp(-model_1.decision_function(X_TEST_1)))
    elif predicting_method == 'probability':
        y_1 = model_1.predict_proba(X_TEST_1)

print('model_2:')
if activation_NN_2 == True:
    tuning_params = []
    score_2 = [0]
    for n in range(hypertraining_iterations):
        ### SETTING RANDOM HYPERPARAMETERS
        hyperparam_epochs = hyperarr_epochs[np.random.randint(0,len(hyperarr_epochs))]
        hyperparam_batchsize = hyperarr_batchsize[np.random.randint(0,len(hyperarr_batchsize))]
        hyperparam_n_layers = hyperarr_n_layers[np.random.randint(0,len(hyperarr_n_layers))]
        hyperparam_start_density = hyperarr_start_density[np.random.randint(0,len(hyperarr_start_density))]
        hyperparam_dropout = hyperarr_dropout[np.random.randint(0,len(hyperarr_dropout))]

        ### GENERATING NN
        model_2 = tf.keras.models.Sequential()
        model_2.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN_2[0]),)))
        for n_layers in range(hyperparam_n_layers):
            model_2.add(tf.keras.layers.Dense(int(hyperparam_start_density/np.power(2,n_layers)), activation='sigmoid'))
            model_2.add(tf.keras.layers.Dropout(hyperparam_dropout))
            model_2.add(tf.keras.layers.BatchNormalization())
        model_2.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        loss_metric_2 = Val_2(X_TRAIN_2_VALIDATION, Y_LABELS_2_VALIDATION)

        model_2.fit(X_TRAIN_2, Y_LABELS_2, batch_size=hyperparam_batchsize, epochs=hyperparam_epochs, callbacks=[loss_metric_2], verbose=1)
        y_validation = model_2.predict(X_TRAIN_2_VALIDATION)
        y_pred_test_2 = model_2.predict(X_TEST_2)

        accuracy = evaluate(2, Y_LABELS_2_VALIDATION, y_validation)
        tuning_params.append([hyperparam_epochs, hyperparam_batchsize, hyperparam_n_layers, hyperparam_start_density, hyperparam_dropout, accuracy])
        print(tuning_params)

        ### WRITING SOLUTION TO SUBMISSION .CSV-FILE
        M_submission_pd = pd.DataFrame(data=y_pred_test_2)
        M_submission_pd.to_csv(r'submission_subtask_2/submission_' + str(n) + '.csv', index=False, header=False)

    ### WRITING PARAMETERS TO FILE 'hyperparam_settings.csv'
    M_hyperparam_settings = pd.DataFrame(data=tuning_params, columns=["epochs", "batch_size", "n_layers_NN", "start_density", "dropout", "accuracy"])
    print(M_hyperparam_settings)
    M_hyperparam_settings.to_csv(r'hyperparam_settings_2.csv')
else:
    model_2 = svm.SVC(kernel='linear', probability=True)
    model_2.fit(X_TRAIN_2, Y_LABELS_2)
    if predicting_method == 'sigmoid':
        y_2 = 1/(1 + np.exp(-model_2.decision_function(X_TEST_2)))
    elif predicting_method == 'probability':
        #y_2 = model_2.predict_proba(X_TEST_2)[:,1]
        y_2 = model_2.predict(X_TEST_2)

print('model_3:')
if activation_NN_3 == True:
    y_3 = NN.neural_network_3(X_TRAIN_3, Y_LABELS_3, X_TEST_3)
else:
    model_3 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)
    model_3.fit(X_TRAIN_3, Y_LABELS_3)
    y_3 = model_3.predict(X_TEST_3)

### Writing to .zip and .csv files
M_submission = np.c_[pid_test, y_pred_test_1, y_pred_test_2, y_3]
M_submission_pd = pd.DataFrame(data=M_submission, columns=["pid","LABEL_BaseExcess","LABEL_Fibrinogen","LABEL_AST","LABEL_Alkalinephos","LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2","LABEL_Bilirubin_direct","LABEL_EtCO2","LABEL_Sepsis","LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"])

M_submission_pd.to_csv(r'sample_' + str(patients) + '_' + str(activation_ntest) + '_' + str(activation_gradient) + '_' + predicting_method + '.csv', index=False, float_format='%.3f')
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample_' + str(patients) + '_' + str(activation_ntest) + '_' + predicting_method + '_' + processing_method + '.zip', index=False, float_format='%.3f', compression=compression_opts)
