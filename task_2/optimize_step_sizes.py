import numpy as np
import pandas as pd
import sklearn.metrics as metrics

#Read in data to train step sizes
magicfactor_testnumber = 0
magicfactor_vitalgradient = 0


VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


def get_score(df_true, df_submission):
    df_submission = df_submission.sort_values('pid')
    df_true = df_true.sort_values('pid')
    task1 = np.mean([metrics.roc_auc_score(df_true[entry], df_submission[entry]) for entry in TESTS])
    task2 = metrics.roc_auc_score(df_true['LABEL_Sepsis'], df_submission['LABEL_Sepsis'])
    task3 = np.mean([0.5 + 0.5 * np.maximum(0, metrics.r2_score(df_true[entry], df_submission[entry])) for entry in VITALS])
    score = np.mean([task1, task2, task3])
    print(task1, task2, task3)
    return score

def read_optimizing_data():
    ### Read in data
    pid_test = pd.read_csv('temporary_predictions/pid_test.csv', delimiter=',')
    y_1_real = pd.read_csv('temporary_predictions/y_1_real.csv', delimiter=',')
    y_1_ntest = pd.read_csv('temporary_predictions/y_1_ntest.csv', delimiter=',')
    y_1_gradient = pd.read_csv('temporary_predictions/y_1_gradient.csv', delimiter=',')
    y_2_real = pd.read_csv('temporary_predictions/y_2_real.csv', delimiter=',')
    y_2_ntest = pd.read_csv('temporary_predictions/y_2_ntest.csv', delimiter=',')
    y_2_gradient = pd.read_csv('temporary_predictions/y_2_gradient.csv', delimiter=',')
    y_3_real = pd.read_csv('temporary_predictions/y_3_real.csv', delimiter=',')
    y_3_ntest = pd.read_csv('temporary_predictions/y_3_ntest.csv', delimiter=',')
    y_3_gradient = pd.read_csv('temporary_predictions/y_3_gradient.csv', delimiter=',')
    return np.asarray(pid_test)[:,1], np.asarray(y_1_real)[:,1:], np.asarray(y_1_ntest)[:,1:], np.asarray(y_1_gradient)[:,1:], np.asarray(y_2_real)[:,1:], np.asarray(y_2_ntest)[:,1:], np.asarray(y_2_gradient)[:,1:], np.asarray(y_3_real)[:,1:], np.asarray(y_3_ntest)[:,1:], np.asarray(y_3_gradient)[:,1:]
    #return np.asarray(pid_test)[:,1], np.asarray(y_1_real)[:,1:], np.asarray(y_2_real)[:,1:], np.asarray(y_3_real)[:,1:]


[pid_test, y_1_real, y_1_ntest, y_1_gradient, y_2_real, y_2_ntest, y_2_gradient, y_3_real, y_3_ntest, y_3_gradient] = read_optimizing_data()


#Moving steps towards ntest and gradient
y_1 = np.add(y_1_real,magicfactor_testnumber*np.subtract(y_1_ntest,y_1_real))   #first step towards ntest
y_2 = np.add(y_2_real,magicfactor_testnumber*np.subtract(y_2_ntest,y_2_real))
y_3 = np.add(y_3_real,magicfactor_testnumber*np.subtract(y_3_ntest,y_3_real))
y_1 = np.add(y_1, magicfactor_vitalgradient*np.subtract(y_1_gradient,y_1))      #second step towards gradient
y_2 = np.add(y_2, magicfactor_vitalgradient*np.subtract(y_2_gradient,y_2))
y_3 = np.add(y_3, magicfactor_vitalgradient*np.subtract(y_3_gradient,y_3))

#Writing to .zip and .csv files
M_submission = np.c_[pid_test, y_1, y_2, y_3]
M_submission_pd = pd.DataFrame(data=M_submission, columns=["pid","LABEL_BaseExcess","LABEL_Fibrinogen","LABEL_AST","LABEL_Alkalinephos","LABEL_Bilirubin_total","LABEL_Lactate","LABEL_TroponinI","LABEL_SaO2","LABEL_Bilirubin_direct","LABEL_EtCO2","LABEL_Sepsis","LABEL_RRate","LABEL_ABPm","LABEL_SpO2","LABEL_Heartrate"])
compression_opts = dict(method='zip', archive_name='sample.csv')
M_submission_pd.to_csv('temporary_predictions/selective_optimizing.zip', index=False, float_format='%.3f', compression=compression_opts)

#EVALUATION
prediction_file = 'temporary_predictions/selective_optimizing.zip'
true_file = 'validation_labels.csv'
df_submission = pd.read_csv(prediction_file)
# generate a baseline based on sample.zip
df_true = pd.read_csv(true_file)                                                #take labels from pid 25067 (half of the dataset) to the end
for label in TESTS + ['LABEL_Sepsis']:
    # round classification labels
    df_true[label] = np.around(df_true[label].values)

print('Score of sample.zip with itself as groundtruth', get_score(df_true, df_submission))
