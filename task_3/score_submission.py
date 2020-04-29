import pandas as pd
import numpy as np
from sklearn.metrics import f1_score



path_predicted = 'sample.csv'
path_true = '../data_3/train_validation.csv'
y_predicted = pd.read_csv(path_predicted, delimiter=',')
y_true = pd.read_csv(path_true, delimiter=',')                                                #take labels from pid 25067 (half of the dataset) to the end
y_true = y_true.iloc[:,1]


print('Score of sample.zip with itself as groundtruth', f1_score(y_true, y_predicted))
