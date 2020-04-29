import numpy as np
import pandas as pd
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
from sklearn import svm


number_mutations = 49999

def read_in_data():
    train_set = pd.read_csv('../data_3/train.csv', delimiter=',')
    train_mutations = train_set.iloc[:,0]
    train_labels = train_set.iloc[:,1]

    test_set = pd.read_csv('../data_3/test.csv', delimiter=',')
    test_mutations = test_set.iloc[:,0]

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

#train model
print('---Started model training---')
model = svm.SVC(kernel='linear',class_weight='balanced')
model.fit(train_mutant, train_labels)
test_labels = model.predict(test_mutant)
print(np.max(test_labels))


#for index in range(len(test_labels)):
#    if test_labels[index] > 0.1:
#        test_labels[index] = 1
#    else:
#        test_labels[index] = 0
#print(test_labels)
#print(np.max(test_labels))

M_submission_pd = pd.DataFrame(data=test_labels)
M_submission_pd.to_csv(r'sample.csv', index=False)
#compression_opts = dict(method='zip', archive_name='sample.csv')
#M_submission_pd.to_csv('sample.zip', index=False, float_format='%.3f', compression=compression_opts)
