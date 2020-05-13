import numpy as np
import pandas as pd
#import yaml                                                                     #for reading running parameters from external yaml file (Euler)
#import argparse
from sklearn import svm
from sklearn.metrics import f1_score
#from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


number_mutations = 49999
c_TP = 0
c_FP = 1
c_FN = 8
c_TN = 0

def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',', nrows=number_mutations)
    train_mutations = train_set.iloc[1:,0]
    train_labels = train_set.iloc[1:,1]

    test_set = pd.read_csv('../data_3/train_validation.csv', delimiter=',')
    test_mutations = test_set.iloc[1:,0]
    test_labels = test_set.iloc[1:,1]

    return train_mutations, train_labels, test_mutations, test_labels

def convert_letters(mutations):
    mutation_number = []
    for row in mutations:
        numbers = []
        for letter in row:
            numbers.append((ord(letter)-64)/25)                                         #'A' = 65, 'Z' = 90 (chronologically)
        mutation_number.append(numbers)
    return mutation_number



#--------------------------------MAIN-------------------------------------------
[train_mutations, train_labels, test_mutations, test_labels] = read_in_data()                #train_mutations and test_mutations contain all the four letters

#convert letters to integers in 4 columns
train_mutant = convert_letters(train_mutations)
test_mutant = convert_letters(test_mutations)

train_mutant = np.asarray(train_mutant)
train_labels = np.asarray(train_labels)
test_mutant = np.asarray(test_mutant)
test_labels = np.asarray(test_labels)


#crossvalidation with score calculation
score = np.zeros(len(arr_tau))
kf = KFold(n_splits=number_of_batches)

for train_index, test_index in kf.split(train_mutant):
    X_train, X_test = train_mutant[train_index], train_mutant[test_index]
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    print('---Started model training---')
    model = LogisticRegression()
    model.fit(X_train, y_train)

model.predict_proba(X_test)
p_0 = model.predict_proba(X_test)[:,0]
p_1 = model.predict_proba(X_test)[:,1]


c_positive = []
c_negative = []
for i in range(len(p_0)):
    c_pos_temp = p_0[i]*c_FP + p_1[i]*c_TP
    c_neg_temp = p_1[i]*c_FN + p_0[i]*c_TN
    c_positive.append(c_pos_temp)
    c_negative.append(c_neg_temp)

y_pred = []
for i in range(len(p_0)):
    if c_positive[i] < c_negative[i]:
        y_temp = 1
    else:
        y_temp = 0
    y_pred.append(y_temp)

#count number of non-zeros
print(np.count_nonzero(y_pred))

#Predict the groundtruth compared to the second half of the training set
print('Score of sample.zip with itself as groundtruth', f1_score(y_test, y_pred))
