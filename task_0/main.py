import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

#The file value is either "train.csv" or "test.csv"
file = 'train.csv'

#read data from csv file and transfer to numpy array
read_train = pd.read_csv('../data_0/' + file, delimiter=',')  # read in train.csv file
data_train = read_train.to_numpy()

#initialize the used arrays
Id = []
y = []
y_pred = []
X = []

#for using the train data
if file == 'train.csv':
    for row in data_train:
        Id.append(int(row[0]))
        y.append(row[1])
        X.append(list(row[2:]))
    for row in X:
        y_pred.append(np.sum(row)/len(row))

    RMSE = mean_squared_error(y, y_pred)**0.5
    print(RMSE)

#for using the test data (submission)
elif file == 'test.csv':
    for row in data_train:
        Id.append(int(row[0]))
        X.append(list(row[1:]))
    for row in X:
        y_pred.append(np.sum(row)/len(row))

#create numpy array for writing
file_prediction = {'Id': Id,
        'y': y_pred
        }
#write Id and y_pred to csv file
dataframe = pd.DataFrame(file_prediction, columns= ['Id', 'y'])
dataframe.to_csv('file_prediction.csv', index=False)
