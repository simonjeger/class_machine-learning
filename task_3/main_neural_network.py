import numpy as np
import pandas as pd
import argparse
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import class_weight

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import normalize, to_categorical
<<<<<<< HEAD


print(tf.version)
=======
>>>>>>> a114b571758c86473c011512a208020c81ae0457

number_mutations = 120000

def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',')
    train_mutations = train_set.iloc[1:,0]
    train_labels = train_set.iloc[1:,1]

    test_set = pd.read_csv('../data_3/test.csv', delimiter=',')
    test_mutations = test_set.iloc[1:,0]

    return train_mutations, train_labels, test_mutations

def convert_letters(mutations):
    mutation_number = []
    for row in mutations:
        numbers = []
        for letter in row:
            numbers.append(ord(letter)-64)                                         #'A' = 65, 'Z' = 90 (chronologically)
        mutation_number.append(numbers)
    return mutation_number

class Metrics(Callback):

    def __init__(self, X, y):
        self.X_val = X
        self.y_val = y
        self.arr_val_f1 = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.round((self.model.predict(self.X_val)))
        val_f1 = f1_score(self.y_val, val_predict)

<<<<<<< HEAD
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.model.save('best_model_kfolds.h5')
            print('------ improoooved ------')
=======
        self.arr_val_f1.append(val_f1)
>>>>>>> a114b571758c86473c011512a208020c81ae0457

        print('— val_f1: %f' %val_f1)
        return

#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[train_mutations, train_labels, test_mutations] = read_in_data()                #train_mutations and test_mutations contain all the four letters

<<<<<<< HEAD
seed(123)
tf.random.set_seed(246)
=======
>>>>>>> a114b571758c86473c011512a208020c81ae0457
#convert letters to integers in 4 columns
train_mutant = convert_letters(train_mutations)
test_mutant = convert_letters(test_mutations)

train_mutant = np.asarray(train_mutant)                                         #Magic stuff to arrange array in nice matrix form:)
train_labels = np.asarray(train_labels)
<<<<<<< HEAD
test_mutant = np.asarray(test_mutant)
test_labels = np.asarray(np.zeros(len(test_mutant)))
print(len(test_labels))

n_folds = 5
kf = KFold(n_splits=n_folds)

for train_index, test_index in kf.split(train_mutant):
    X_train, X_val = train_mutant[train_index], train_mutant[test_index]
    y_train, y_val = train_labels[train_index], train_labels[test_index]
    #X_train = normalize(X_train, axis=1)
    #X_val = normalize(X_val, axis=1)
    #y_train = to_categorical(y_train)
    #y_val = to_categorical(y_val)
    print('---Started model training---')
=======

X_train, X_val, y_train, y_val = train_test_split(train_mutant, train_labels, test_size=0.1, random_state=42)
X_train = normalize(X_train, axis=1)
X_val = normalize(X_val, axis=1)

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights = dict(enumerate(class_weights))

#y_train = to_categorical(y_train)
#y_val = to_categorical(y_val)

#Final model with the optimal threshold value determined before
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(4,)))
model.add(tf.keras.layers.Dense(190, activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(95, activation='sigmoid'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
>>>>>>> a114b571758c86473c011512a208020c81ae0457

    #Final model with the optimal threshold value determined before
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(84,)))
    model.add(tf.keras.layers.Dense(196, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(98, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Dense(80, activation='tanh'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

<<<<<<< HEAD
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    f1_metric = Val_F1(X_val,y_val)

    weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    weights = dict(enumerate(weights))

    model.fit(X_train, y_train,validation_data=(X_val,y_val), epochs=30, class_weight=weights, callbacks=[f1_metric],verbose=1)

    #test_mutant = normalize(test_mutant, axis=1)
    final_model = load_model('best_model_kfolds.h5')

    print('predictions')
    test_labels += final_model.predict(test_mutant).reshape(-1)
    tf.keras.backend.clear_session()


print('rounding')
test_labels = np.round(np.divide(test_labels,n_folds))

print('write csv file')
np.savetxt('sample_neural_kfolds.csv', test_labels, delimiter='\n', fmt ='%d')
#M_submission_pd = pd.DataFrame(data=test_labels)
#M_submission_pd.to_csv(r'sample_neural.csv', index=False)
=======
metrics = Metrics(X_val,y_val)

model.fit(X_train, y_train, batch_size=32, epochs=15, callbacks=[metrics], verbose=1,class_weight=class_weights)

test_mutant = normalize(test_mutant, axis=1)
test_labels = model.predict_classes(test_mutant)

M_submission_pd = pd.DataFrame(data=test_labels)
M_submission_pd.to_csv(r'sample_neural.csv', index=False)
>>>>>>> a114b571758c86473c011512a208020c81ae0457
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
