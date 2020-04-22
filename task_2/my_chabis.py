import numpy as np
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeCV
from sklearn import svm


def writing_prediction_files(pid_test, y_1_real, y_1_ntest, y_2_real, y_2_ntest, y_3_real):
    file_pid_test = pd.DataFrame(data=pid_test)
    file_pid_test.to_csv('temporary_predictions_1500/pid_test.csv')
    file_y_1_real = pd.DataFrame(data=y_1_real)
    file_y_1_real.to_csv('temporary_predictions_1500/y_1_real.csv')
    file_y_1_ntest = pd.DataFrame(data=y_1_ntest)
    file_y_1_ntest.to_csv('temporary_predictions_1500/y_1_ntest.csv')
    #file_y_1_gradient = pd.DataFrame(data=y_1_gradient)
    #file_y_1_gradient.to_csv('temporary_predictions_1500/y_1_gradient.csv')
    file_y_2_real = pd.DataFrame(data=y_2_real)
    file_y_2_real.to_csv('temporary_predictions_1500/y_2_real.csv')
    file_y_2_ntest = pd.DataFrame(data=y_2_ntest)
    file_y_2_ntest.to_csv('temporary_predictions_1500/y_2_ntest.csv')
    #file_y_2_gradient = pd.DataFrame(data=y_2_gradient)
    #file_y_2_gradient.to_csv('temporary_predictions_1500/y_2_gradient.csv')
    file_y_3_real = pd.DataFrame(data=y_3_real)
    file_y_3_real.to_csv('temporary_predictions_1500/y_3_real.csv')
    file_y_3_ntest = pd.DataFrame(data=y_3_ntest)
    file_y_3_ntest.to_csv('temporary_predictions_1500/y_3_ntest.csv')
    #file_y_3_gradient = pd.DataFrame(data=y_3_gradient)
    #file_y_3_gradient.to_csv('temporary_predictions_1500/y_3_gradient.csv')
    return True
