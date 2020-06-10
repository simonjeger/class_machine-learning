import numpy as np
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn import svm
from sklearn import preprocessing
import sklearn.metrics as metrics

def read_in():
    df_train_labels = pd.read_csv('../data_2/train_labels.csv')             #Y_train
    df_train_features = pd.read_csv('../data_2/train_features.csv')         #X_train
    df_test_features = pd.read_csv('../data_2/test_features.csv')           #X_test

    X_train = pd.concat([df_train_features[df_train_features.columns[0]],       # select important columns
                        df_train_features[df_train_features.columns[2]],
                        df_train_features[df_train_features.columns[7]],
                        df_train_features[df_train_features.columns[11]],
                        df_train_features[df_train_features.columns[22]],
                        df_train_features[df_train_features.columns[25]],
                        df_train_features[df_train_features.columns[28]],
                        df_train_features[df_train_features.columns[32]],
                        df_train_features[df_train_features.columns[35]]],axis=1)

    X_test = pd.concat([df_test_features[df_test_features.columns[0]],       # select important columns
                        df_test_features[df_test_features.columns[2]],
                        df_test_features[df_test_features.columns[7]],
                        df_test_features[df_test_features.columns[11]],
                        df_test_features[df_test_features.columns[22]],
                        df_test_features[df_test_features.columns[25]],
                        df_test_features[df_test_features.columns[28]],
                        df_test_features[df_test_features.columns[32]],
                        df_test_features[df_test_features.columns[35]]],axis=1)

    return X_train, X_test, df_train_labels, df_train_features, df_test_features

def preprocessing_task_1(X):
    df = X_mean.groupby( ['pid'] )
    df = 
    return X.notnull()*1

def number_of_tests(X):
    X = X.drop(['pid'], axis=1)
    X = X.drop(['Age'], axis=1)
    return X.notnull()*1

def remove_nan(X, file_name):
    df_nanless = pd.DataFrame() #indx new array for mean value

    for i in range (0,int(X.index.size/12)):
        print('prepare ' + file_name + ': patiend nr. ' + str(i))

        X_mean = X.loc[0+i*12:11+i*12]      #select the 12 rows coresponding to one patient
        df = X_mean.groupby( ['pid'] ).mean()     #mean all the values
        df_nanless = df_nanless.append(df)    #append the new values

    df_nanless.fillna(X.mean(), inplace=True)    #replace NaN with mean Values

    Patient_data_mean = pd.DataFrame(df_nanless)
    Patient_data_mean.to_csv('../data_2/' + file_name + '.csv') #store data to save time

def normalize_pandas(df):
    return (df-df.mean())/df.std()

def support_vector_machine(task, column, X_train_prep, Y_train_prep, X_test_prep):
    Y_test_final = X_test_prep[X_test_prep.columns[0]]                                   #initializing the size and type of my final results
    for x in range (column[0],column[1]):                                               #for loop to calculate each column of Y_test
        # predict with training set
        Y_train = Y_train_prep[Y_train_prep.columns[x]]                                 #select corespondig column from Y_train
        X_train_temp, X_validation, Y_train_temp, Y_validation = train_test_split(X_train_prep,Y_train, test_size = 0.2)

        model = svm.SVC(kernel='rbf', probability=True)
        model.fit(X_train_temp, Y_train_temp)

        # evaluate with training set
        Y_pred = model.predict_proba(X_validation)
        df = pd.DataFrame(data = Y_pred[:,1])
        print('accuracy column '+ str(x) + ': ' + str(evaluate(task, Y_validation, df)))

        # predict with test set
        Y_test_temp = model.predict_proba(X_test_prep)
        Y_test_temp = pd.DataFrame(data=Y_test_temp[:,1], columns = [column_names[x-1]])
        Y_test_final = pd.concat([Y_test_final,Y_test_temp],axis=1)
    return Y_test_final, Y_validation, df

def logistic_regression(task, column, X_train_prep, Y_train_prep, X_test_prep):
    Y_test_final = X_test_prep[X_test_prep.columns[0]]                                   #initializing the size and type of my final results
    for x in range (column[0],column[1]):                                               #for loop to calculate each column of Y_test
        # predict with training set
        Y_train = Y_train_prep[Y_train_prep.columns[x]]                                 #select corespondig column from Y_train
        X_train_temp, X_validation, Y_train_temp, Y_validation = train_test_split(X_train_prep,Y_train, test_size = 0.2)

        model = LogisticRegression()
        model.fit(X_train_temp, Y_train_temp)

        # evaluate with training set
        Y_pred = model.predict_proba(X_validation)
        df = pd.DataFrame(data = Y_pred[:,1])
        print('accuracy column '+ str(x) + ': ' + str(evaluate(task, Y_validation, df)))

        # predict with test set
        Y_test_temp = model.predict_proba(X_test_prep)
        Y_test_temp = pd.DataFrame(data=Y_test_temp[:,1], columns = [column_names[x-1]])
        Y_test_final = pd.concat([Y_test_final,Y_test_temp],axis=1)
    return Y_test_final, Y_validation, df

def ridge_regression(task, column, X_train_prep, Y_train_prep, X_test_prep):
    Y_test_final = X_test_prep[X_test_prep.columns[0]]                                   #initializing the size and type of my final results
    for x in range (column[0],column[1]):                                               #for loop to calculate each column of Y_test
        # predict with training set
        Y_train = Y_train_prep[Y_train_prep.columns[x]]                                 #select corespondig column from Y_train
        X_train_temp, X_validation, Y_train_temp, Y_validation = train_test_split(X_train_prep,Y_train, test_size = 0.2)

        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1], cv=None)
        model.fit(X_train_temp, Y_train_temp)

        # evaluate with training set
        Y_pred = model.predict(X_validation)
        df = pd.DataFrame(data = Y_pred)
        print('accuracy column '+ str(x) + ': ' + str(evaluate(task, Y_validation, df)))

        # predict with test set
        Y_test_temp = model.predict(X_test_prep)
        Y_test_temp = pd.DataFrame(data=Y_test_temp, columns = [column_names[x-1]])
        Y_test_final = pd.concat([Y_test_final,Y_test_temp],axis=1)
    return Y_test_final, Y_validation, df

def evaluate(task, Y_validation, df):
    if task == 1:
        return metrics.roc_auc_score(Y_validation, df)
    if task == 2:
        return metrics.roc_auc_score(Y_validation, df)
    if task == 3:
        return 0.5 + 0.5 * np.maximum(0, metrics.r2_score(Y_validation, df))

#-------------------------------------------
##Read in data (optional if it has been done once before)

[X_train, X_test, df_train_labels, df_train_features, df_test_features] = read_in()

#-----------------------------------------------

# normalize_pandas
X_train = normalize_pandas(X_train)
X_test = normalize_pandas(X_test)

# add "number of tests" to task 1 & 2
#X_train = pd.concat([X_train, number_of_tests(X_train)], axis=1, sort=False)
#X_test = pd.concat([X_test, number_of_tests(X_test)], axis=1, sort=False)

#print(number_of_tests(X_train))

# remove_nan

preprocessing_task_1(X_train)

remove_nan(X_train, "patient_data_train_prep")
remove_nan(X_test, "patient_data_test_prep")
#-----------------------------------------------

X_train_prep = pd.read_csv("../data_2/patient_data_train_prep.csv")             #Read in previously generated data
Y_train_prep = df_train_labels
X_test_prep = pd.read_csv("../data_2/patient_data_test_prep.csv")

X_train_withoutpid = X_train_prep[X_train_prep.columns[1:9]]     #Remove Pid column
X_test_withoutpid = X_test_prep[X_test_prep.columns[1:9]]

pid = X_test_prep[X_test_prep.columns[0]]

column_names = ['LABEL_BaseExcess','LABEL_Fibrinogen',
               'LABEL_AST','LABEL_Alkalinephos',
               'LABEL_Bilirubin_total','LABEL_Lactate',
               'LABEL_TroponinI','LABEL_SaO2',
               'LABEL_Bilirubin_direct','LABEL_EtCO2',
              'LABEL_Sepsis','LABEL_RRate',
              'LABEL_ABPm','LABEL_SpO2',
              'LABEL_Heartrate']

#-------------------------------------------

n = 1000
X_train_prep = X_train_prep.head(n)
X_test_prep = X_test_prep.head(n)
Y_train_prep = Y_train_prep.head(n)

##Solve task 1
Y_test_final_1 = support_vector_machine(1, [1,11], X_train_prep, Y_train_prep, X_test_prep)

#-------------------------------------------
##Solve task 2
Y_test_final_2 = logistic_regression(2, [11,12], X_train_prep, Y_train_prep, X_test_prep)

#-------------------------------------------
##Solve task 3
Y_test_final_3 = ridge_regression(3, [12,16], X_train_prep, Y_train_prep, X_test_prep)
