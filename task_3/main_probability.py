import numpy as np
import pandas as pd
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


number_mutations = 50000                                                        #maximum: 112000
number_of_batches = 5
arr_threshold = [0.005, 0.008, 0.01, 0.02, 0.03, 0.04]


def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',', nrows=number_mutations)
    train_mutations = train_set.iloc[1:number_mutations,0]
    train_labels = train_set.iloc[1:number_mutations,1]

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

def probability_threshold(y_test, y_test_predicted, threshold):
    y_threshold = np.zeros(len(y_test_predicted))
    for index in range(len(y_threshold)):
        if y_test_predicted[index] > threshold:
            y_threshold[index] = 1
    score = f1_score(y_test, y_threshold)
    return score

#--------------------------------MAIN-------------------------------------------
### Read in the whole data set
[train_mutations, train_labels, test_mutations] = read_in_data()                #train_mutations and test_mutations contain all the four letters

#convert letters to integers in 4 columns
train_mutant = convert_letters(train_mutations)
test_mutant = convert_letters(test_mutations)
np.asarray(train_mutant)
np.asarray(test_mutant)

#crossvalidation with score calculation
score = np.zeros(len(arr_threshold))
kf = KFold(n_splits=number_of_batches)
train_mutant = np.asarray(train_mutant)                                         #Magic stuff to arrange array in nice matrix form:)
test_mutant = np.asarray(test_mutant)
train_labels = np.asarray(train_labels)

for train_index, test_index in kf.split(train_mutant):
    X_train, X_test = train_mutant[train_index], train_mutant[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    print('---Started model training---')
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_test_predicted = model.predict_proba(X_test)[:,1]
    for index_threshold in arr_threshold:
        score[arr_threshold.index(index_threshold)] += probability_threshold(y_test, y_test_predicted, index_threshold)
    del y_test_predicted
    del model
print(score)

#Final model with the optimal threshold value determined before
print('---FINAL MODEL---')
final_threshold = arr_threshold[np.argmax(score)]
print(final_threshold)
model_final = svm.SVC(kernel='linear', probability=True)
model_final.fit(train_mutant, train_labels)
test_labels = model_final.predict_proba(test_mutant)[:,1]
for index in range(len(test_labels)):
    if test_labels[index] > final_threshold:
        test_labels[index] = 1
    else:
        test_labels[index] = 0


M_submission_pd = pd.DataFrame(data=test_labels)
M_submission_pd.to_csv(r'sample.csv', index=False)
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
