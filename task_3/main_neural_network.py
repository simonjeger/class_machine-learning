import numpy as np
import pandas as pd
import argparse
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from numpy.random import seed
import numpy.random as random
np.random.seed(300)
# TensorFlow and tf.keras
import keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import SGD
from keras.callbacks import Callback,ModelCheckpoint
from keras.layers import BatchNormalization
from keras.utils import normalize, to_categorical
from tensorflow import random
import keras.backend as K

import tensorflow as tf
tf.random.set_seed(246)

number_mutations = 120000

def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',')
    train_mutations = train_set['Sequence']
    train_labels = train_set['Active']

    test_set = pd.read_csv('../data_3/test.csv', delimiter=',')
    test_mutations = test_set['Sequence']

    return train_mutations, train_labels, test_mutations

def convert_letters(mutations):
    mutation_number = []
    for row in mutations:
        numbers = []
        for letter in row:
            numbers.append(ord(letter)-64)                                      #'A' = 65, 'Z' = 90 (chronologically)
        mutation_number.append(numbers)
    return mutation_number

def convert_letters_alternative(mutations):
    letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mutation_number = []

    for i in mutations:
        letter_indicators = np.zeros(4*len(letters))
        n = 0
        for j in list(i):
            for k in letters:
                letter_indicators[n] = int(j == k)
                n = n+1
        mutation_number.append(letter_indicators)
    return mutation_number

def convert_letters_onehotencoding(mutations):
    letters = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mutation_number = []

    for row in mutations:
        numbers = []
        for letter in row:
            numbers.append(letter)                                      #'A' = 65, 'Z' = 90 (chronologically)
        mutation_number.append(numbers)
    cat = OneHotEncoder()
    data = cat.fit_transform(mutation_number).toarray()

    print(len(data))
    return data

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class Val_F1(Callback):

    def __init__(self, X, y):
        self.X_val = X
        self.y_val = y
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.round(self.model.predict(self.X_val).reshape(-1))
        val_f1 = f1_score(self.y_val, val_predict)

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            print('%i — val_f1: %f -- improved' %(epoch,val_f1))
        else:
            print('%i — val_f1: %f' %(epoch,val_f1))

        if val_f1 > score[0]:
            score[0] = val_f1
            best_variables = [dropout_ratio_1, dropout_ratio_2, density_1, train_test_ratio, score[0]]
            print('----------- new best variables -----------')
            print('— dropout_ratio_1: %f' %best_variables[0])
            print('— dropout_ratio_2: %f' %best_variables[1])
            print('— density_1: %f' %best_variables[2])
            print('— train_test_ratio: %f' %best_variables[3])
            print('— best score: %f' %score[0])
            self.model.save('best_model/best_model_'+ str(best_variables[0])+'_'+ str(best_variables[1])+'_'+ str(best_variables[2])+'_'+str(best_variables[3])+'_'+ '.h5')
            np.savetxt('best_variables/best_variables_'+ str(best_variables[0])+'_'+ str(best_variables[1])+'_'+ str(best_variables[2])+'_'+str(best_variables[3])+'_'+ '.csv', best_variables, delimiter='\n', fmt ='%10.5f')
            #print('predictions')
            test_labels = self.model.predict(test_mutant).reshape(-1)
            #print('rounding')
            test_labels = np.round(test_labels)

            #print('write csv file')
            np.savetxt('Prediction/sample_neural_'+ str(best_variables[0])+'_'+ str(best_variables[1])+'_'+ str(best_variables[2])+'_'+str(best_variables[3])+'_'+ '.csv', test_labels, delimiter='\n', fmt ='%d')

def create_model(dropout_ratio_1, dropout_ratio_2, density_1):
    density_2 = int(np.round(np.divide(density_1,2)))
    #Final model with the optimal threshold value determined before
    model = Sequential()
    model.add(Dense(density_1, activation='relu'))
    model.add(Dropout(dropout_ratio_1))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(density_2, activation='relu'))
    model.add(Dropout(dropout_ratio_2))
    #model.add(tf.keras.layers.Dense(80, activation='tanh'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    return model
#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[train_mutations, train_labels, test_mutations] = read_in_data()                #train_mutations and test_mutations contain all the four letters

#convert letters to integers in 4 columns
train_mutant = convert_letters_onehotencoding(train_mutations)
test_mutant = convert_letters_onehotencoding(test_mutations)

train_mutant = np.asarray(train_mutant)                                         #Magic stuff to arrange array in nice matrix form:)
train_labels = np.asarray(train_labels)
test_mutant = np.asarray(test_mutant)
test_labels = np.asarray(np.zeros(len(test_mutant)))

epoch = 2000
hypertuning = False

score = [0]
best_variables = []

if hypertuning == True:
    Dropout_ratio_list_1 = [0.2, 0.3]
    Dropout_ratio_list_2 = [0.2, 0.3]

    Density_1_list = [144, 168, 188, 192, 202]
    train_test_ratio_list = [0.1, 0.12, 0.15]
else:
    Dropout_ratio_list_1 = [0.2]
    Dropout_ratio_list_2 = [0.2]
    Density_1_list = [144]
    train_test_ratio_list = [0.1]
for i in range(1,25):
    seed(i)
    train_test_ratio = np.random.choice(train_test_ratio_list)
    dropout_ratio_1 = np.random.choice(Dropout_ratio_list_1)
    seed(2*i)
    dropout_ratio_2 = np.random.choice(Dropout_ratio_list_2)
    density_1 = np.random.choice(Density_1_list)
    X_train, X_val, y_train, y_val = train_test_split(train_mutant, train_labels, test_size = train_test_ratio, random_state = 42)

    #print('---Started model training---')

    density_2 = int(np.round(np.divide(density_1,2)))
    print(i)
    print('***************** Hyperparameters: *****************')

    print(dropout_ratio_1, dropout_ratio_2, density_1, train_test_ratio)

    model = create_model(dropout_ratio_1, dropout_ratio_2, density_1)

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    #print('---model created---')

    f1_metric = Val_F1(X_val,y_val)


    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    weights = dict(enumerate(weights))

    #print('---model fitting---')

    model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size = 32, epochs = epoch, class_weight=weights, callbacks=[f1_metric],verbose=0)
    #model.summary()

#test_mutant = normalize(test_mutant, axis=1)
#final_model = load_model('best_model'+ str(best_variables[0])+'_'+ str(best_variables[1])+'_'+ str(best_variables[2])+'_'+str(best_variables[3])+'_'+ '.h5')


#print('predictions')
#test_labels = final_model.predict(test_mutant).reshape(-1)
#keras.backend.clear_session()


#print('rounding')
#test_labels = np.round(test_labels)

#print('write csv file')
#np.savetxt('sample_neural_5.csv', test_labels, delimiter='\n', fmt ='%d')
#M_submission_pd = pd.DataFrame(data=test_labels)
#M_submission_pd.to_csv(r'sample_neural.csv', index=False)
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
