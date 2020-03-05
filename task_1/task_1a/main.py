import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

#The file value is either "train.csv" or "test.csv"
file = 'train.csv'

#read data from csv file and transfer to numpy array
read_train = pd.read_csv('../../data_1/data_1a' + file, delimiter=',')  # read in train.csv file
data_train = read_train.to_numpy()

#initialize the used arrays
Id = []
y = []
y_pred = []
X = []

for row in data_train:
    Id.append(int(row[0]))
    y.append(row[1])
    X.append(list(row[2:]))
