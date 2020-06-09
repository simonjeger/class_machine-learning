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
def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',')
    train_mutations = train_set['Sequence']
    train_labels = train_set['Active']

    test_set = pd.read_csv('../data_3/test.csv', delimiter=',')
    test_mutations = test_set['Sequence']

    return train_mutations, train_labels, test_mutations

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
#test_mutant = normalize(test_mutant, axis=1)
final_model = load_model('best_model_5.h5')


print('predictions')
test_labels = final_model.predict(test_mutant).reshape(-1)


print('rounding')
test_labels = np.round(test_labels)

print('write csv file')
np.savetxt('sample_prediction.csv', test_labels, delimiter='\n', fmt ='%d')
#M_submission_pd = pd.DataFrame(data=test_labels)
#M_submission_pd.to_csv(r'sample_neural.csv', index=False)
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
