import numpy as np
import pandas as pd

activation_hack_training_data = False                                           #decide whether new training data should be estimated according to most frequent predictions
n_submission_files = 20
submission_file_name = [5, 19, 0, 4, 9, 17, 18, 11, 7]

def read_submission_data():
    for i in submission_file_name:
        name = 'submission_' + str(i) + '.csv'
        new_submission_file = pd.read_csv('submission_files/' + name, header=None)
        if submission_file_name.index(i)==0:
            submission_total = pd.DataFrame(data=new_submission_file)
        else:
            submission_total = pd.concat([submission_total, new_submission_file], axis=1)
    return submission_total

#--------------------------------MAIN-------------------------------------------
#READ DATA OF ALL SINGLE SUBMISSION FILES
submission_total = read_submission_data()
print(submission_total)
#CALCULATE THE ROW-WISE SUM OF THE SUBMISSION MATRIX
y_sum = submission_total.sum(axis=1)

#COUNT HOW MANY TIMES EACH IMAGE TRIPLE HAS BEEN PREDICTED AS TRUE
print('frequency of counted ones:')
y_sum_counts = y_sum.value_counts()
y_sum_counts = y_sum_counts.sort_index()
print(y_sum_counts)

#PREDICT FINAL CHOICE ACCORDING TO OCCURANCE FREQUENCY
prediction_threshhold = len(submission_total.columns)/2
print('prediction threshold: ', prediction_threshhold)
y_prediction = []
for i in range(len(y_sum)):
    if y_sum[i] > prediction_threshhold:
        y_prediction.append(1)
    elif y_sum[i] < prediction_threshhold:
        y_prediction.append(0)
    else:
        y_prediction.append(submission_total.iloc[i,0])                         #take the value of the most important one (first column)

#CREATE THE FINAL PREDICTION FILE
M_submission_pd = pd.DataFrame(data=y_prediction)
M_submission_pd.to_csv(r'submission_hack.csv', index=False, header=False)


if activation_hack_training_data:
    #FIND INDICES OF THOSE TRIPLETS THAT HAVE BEEN PREDICTED ALWAYS TRUE OR ALWAYS FALSE
    index_true = y_sum[y_sum==len(submission_total.columns)].index.values.astype(int)
    index_false = y_sum[y_sum==0].index.values.astype(int)
    test_triplets = pd.read_csv('../data_4/test_triplets.txt', delimiter=' ', header=None)
    hack_true_triplets, hack_false_triplets, hack_train_triplets = [], [], []

    #EXTRACT ALL TRIPLETS THAT ARE CERTAINLY TRUE
    for i_true in range(len(index_true)):
        hack_true_triplets.append(test_triplets.iloc[index_true[i_true],:])
    hack_true_triplets = np.asarray(hack_true_triplets)
    for i_false in range(len(index_false)):
        hack_false_triplets.append(test_triplets.iloc[index_false[i_false],:])
    hack_false_triplets = np.asarray(hack_false_triplets)

    #SWAP THE SECOND AND THE THIRD TO MAKE THE TRIPLETS TRUE AGAIN
    hack_false_triplets = np.c_[hack_false_triplets[:,0], hack_false_triplets[:,2], hack_false_triplets[:,1]]
    #CONCATENATE THE TWO (TRUE AND FALSE) CERTAIN MATRICES WITH THE TRIPLETS
    new_train_triplets = np.concatenate((hack_true_triplets, hack_false_triplets))
    #SAVE THE NEWLY FOUND CERTAIN TRIPLETS IN 'hack_train_triplets' FOR TRAINING IN THE NEXT ITERATION
    np.savetxt(r'../data_4/hack_train_triplets.txt', new_train_triplets, delimiter=',',  fmt='%5.0d')
