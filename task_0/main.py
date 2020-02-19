import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

file = 'test.csv'

read_train = pd.read_csv('../data_0/' + file, delimiter=',')  # read in train.csv file
data_train = read_train.to_numpy()

id = []
y = []
y_pred = []
X = []

if file == 'train.csv':
    for row in data_train:
        id.append(int(row[0]))
        y.append(row[1])
        X.append(list(row[2:]))

    RMSE = mean_squared_error(y, y_pred)**0.5
    print(RMSE)

if file == 'test.csv':
    for row in data_train:
        id.append(int(row[0]))
        X.append(list(row[1:]))

for row in X:
    y_pred.append(np.sum(row)/len(row))

file_prediction = {'id': id,
        'y_pred': y_pred
        }
dataframe = pd.DataFrame(file_prediction, columns= ['id', 'y_pred'])
dataframe.to_csv('file_prediction.csv', index=False)
