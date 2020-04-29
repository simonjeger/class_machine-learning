import numpy as np
import pandas as pd
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


number_mutations = 111999
number_of_batches = 8
arr_tau = [0.5, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',', nrows=number_mutations)
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
            numbers.append(ord(letter))                                         #'A' = 65, 'Z' = 90 (chronologically)
        mutation_number.append(numbers)
    return mutation_number
#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[train_mutations, train_labels, test_mutations] = read_in_data()                #train_mutations and test_mutations contain all the four letters

#convert letters to integers in 4 columns
train_mutant = convert_letters(train_mutations)
test_mutant = convert_letters(test_mutations)

#crossvalidation with score calculation
score = np.zeros(len(arr_tau))
kf = KFold(n_splits=number_of_batches)
train_mutant = np.asarray(train_mutant)                                         #Magic stuff to arrange array in nice matrix form:)
train_labels = np.asarray(train_labels)

for train_index, test_index in kf.split(train_mutant):
    X_train, X_test = train_mutant[train_index], train_mutant[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    print('---Started model training---')
    model = svm.SVC(kernel='rbf', class_weight='balanced')
    model.fit(X_train, y_train)

    for index_tau in arr_tau:
        y_test_predicted = np.sign(model.decision_function(X_test)-index_tau)
        y_test_predicted = np.where(y_test_predicted==-1.0, 0.0, y_test_predicted)
        y_test_predicted = np.asarray(y_test_predicted)
        print(np.count_nonzero(y_test_predicted))

        score[arr_tau.index(index_tau)] += f1_score(y_test, y_test_predicted)
    del y_test_predicted
    del model
score = np.divide(score, number_of_batches)
print(score)

#Final model with the optimal threshold value determined before
print('---FINAL MODEL---')
final_tau = arr_tau[np.argmax(score)]
print(final_tau)
model_final = svm.SVC(kernel='rbf', class_weight='balanced')
model_final.fit(train_mutant, train_labels)
test_labels = np.sign(model_final.decision_function(test_mutant)-final_tau)
test_labels = np.where(test_labels==-1.0, 0.0, test_labels)


M_submission_pd = pd.DataFrame(data=test_labels)
M_submission_pd.to_csv(r'sample.csv', index=False)
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
