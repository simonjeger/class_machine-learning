import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import pandas as pd

#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data():
    read_train_features = pd.read_csv('../task_2/sample_try.csv', delimiter=';')
    read_train_features = read_train_features.replace('nan', np.NaN)

    read_train_features.fillna(read_train_features.mean(), inplace=True)

    data_train_features = read_train_features.to_numpy()

    X_FEATURES = []

    for row in data_train_features:
        X_FEATURES.append(list(row[1:]))

    return X_FEATURES, data_train_features



print(read_in_data())

old_array = [1,2,3,4,5,6,7,8,9,10]
n = 2
new_array = []
for j in range(len(old_array)):
    for i in range(n):
        new_array.append(old_array[j])
