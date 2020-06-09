import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV,LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_selection import RFE


#READ DATA FROM CSV FILE AND TRANSFER TO NUMPY ARRAY
def read_in_data(file):

    read_train = pd.read_csv('../../data_1/data_1b/' + file, delimiter=',')
    data_train = read_train.to_numpy()

    Id = []                                                                     #initialize the used arrays
    y = []
    X_lin = []

    for row in data_train:                                                      #Assign "Id", "y" and "X"
        Id.append(int(row[0]))
        y.append(row[1])
        X_lin.append(list(row[2:]))
    #print(X_lin)
    return [Id,y,X_lin]

def add_features(X_lin):
    #The matrix X must have as many columns as number_of_features
    phi_6, phi_7, phi_8, phi_9, phi_10 = [], [], [], [], []                     #for quadratic features
    phi_11, phi_12, phi_13, phi_14, phi_15 = [], [], [], [], []                 #for exponential features
    phi_16, phi_17, phi_18, phi_19, phi_20 = [], [], [], [], []                 #for cosine features
    phi_21 = []                                                                 #for constant features


    for row in X_lin:
        phi_6.append(row[0]**2)
        phi_7.append(row[1]**2)
        phi_8.append(row[2]**2)
        phi_9.append(row[3]**2)
        phi_10.append(row[4]**2)
        phi_11.append(np.exp(row[0]))
        phi_12.append(np.exp(row[1]))
        phi_13.append(np.exp(row[2]))
        phi_14.append(np.exp(row[3]))
        phi_15.append(np.exp(row[4]))
        phi_16.append(np.cos(row[0]))
        phi_17.append(np.cos(row[1]))
        phi_18.append(np.cos(row[2]))
        phi_19.append(np.cos(row[3]))
        phi_20.append(np.cos(row[4]))
        phi_21.append(1)

    X = np.c_[X_lin, phi_6, phi_7, phi_8, phi_9, phi_10, phi_11, phi_12, phi_13, phi_14, phi_15, phi_16, phi_17, phi_18, phi_19, phi_20, phi_21]
    return X

#EXACT SOLUTION OF THE LINEAR REGRESSION
def ridge_regression(X, y, _lambda):
    if len(_lambda)==1:
        clf = Ridge(alpha = _lambda, fit_intercept = False)
        clf.fit(X,y)
    else:
        clf = RidgeCV(alphas = _lambda, fit_intercept = False)
        clf.fit(X,y)
        print('alpha: %f' %clf.alpha_)
    w_star = clf.coef_

    return w_star

def lasso_regression(X, y, _lambda):
    clf = LassoCV(alphas = _lambda, fit_intercept = False,max_iter= 100000)
    clf.fit(X,y)
    w_star = clf.coef_
    return w_star

def error_predict(X_val, y_val, weights):
    y_pred = np.dot(weights,X_val.T)
    error = mean_squared_error(y_val,y_pred)
    return error

def scaling_data(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled

def feature_selection(X_train, y_train,X_val, chosen_penalty):
    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(y_train)
    #sel_ = SelectFromModel(LogisticRegression(C=0.0001, penalty=chosen_penalty))
    sel_ = SelectFromModel(Ridge(alpha = 100))

    sel_.fit(X_train, training_scores_encoded)
    X_train_selected = sel_.transform(X_train)
    X_val_selected = sel_.transform(X_val)

    return sel_.get_support(), X_train_selected, X_val_selected

def manual_feature_selection(w_star,X_train,X_val, chosen_treshhold):
    selected_features = np.zeros(number_of_features, dtype=bool)
    for i in range(number_of_features-1,-1,-1):
        if abs(w_star[i]) < chosen_treshhold :
            X_train = np.delete(X_train, i, 1)
            X_val = np.delete(X_val, i, 1)

        else:
            selected_features[i] = True
    return selected_features, X_train, X_val

def feature_scaling(X_train,y_train,amount):
    model = Ridge(alpha = 100)
    rfe = RFE(model, amount)
    fit = rfe.fit(X_train,y_train)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    selected_features = fit.support_
    return selected_features

def leave_out_feature(X_train,X_val,selected_features):
    for i in range(len(selected_features)):
        if selected_features[i] == False:
            X_train = np.delete(X_train, i, 1)
            X_val = np.delete(X_val, i, 1)
    return X_train, X_val
###MAIN-------------------------------------------------------------------------
[Id, y, X_lin] = read_in_data('train.csv')                                      #Read the data from file
number_of_features = 21
_lambda = [1,5,10,20,50,100, 200, 300, 1000]
_lambda = [1,10,100, 1000]
#_lambda = [100]
activation_scaling = False
activation_feature_selection = 'greedy_backwards'
chosen_penalty = 'l2'

file = open("weights.csv", "w")                                                 #Create submission file with writing access

X = add_features(X_lin)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, random_state = 42)

if activation_scaling == True:
    print('scaling')
    X_train = scaling_data(X_train)

if activation_feature_selection == 'True':
    selected_features, X_train, X_val = feature_selection(X_train, y_train,X_val,chosen_penalty)
elif activation_feature_selection == 'Manual':
    w_star = ridge_regression(X_train, y_train, _lambda)
    print(w_star)
    selected_features, X_train, X_val = manual_feature_selection(w_star,X_train,X_val, 0.01)
elif activation_feature_selection == 'greedy_backwards':
    w_star_ridge = ridge_regression(X_train, y_train, _lambda)
    error_ridge = error_predict(X_val, y_val, w_star_ridge)
    error_new = 0
    selected_features = np.ones(number_of_features, dtype=bool)
    left_out_indices = []
    amount = 20
    for i in range(number_of_features,0,-1):
        selected_features_iterative = feature_scaling(X_train,y_train, amount)
        X_train_selected, X_val_selected = leave_out_feature(X_train,X_val,selected_features_iterative)
        w_star_new = ridge_regression(X_train_selected, y_train, _lambda)
        error_new = error_predict(X_val_selected, y_val, w_star_new)
        if error_new < error_ridge:
            w_star_ridge = w_star_new
            error_ridge = error_new
            X_train = X_train_selected
            X_val = X_val_selected
            amount -=1
            for m in range(len(selected_features_iterative)):
                if selected_features_iterative[m] == False:
                    print('left_out_indices:')

                    if not left_out_indices:
                        print('first left out')
                        left_out_indices = [m]
                    else:
                        print('# DEBUG: ')
                        print(left_out_indices[number_of_features-i-1])
                        if left_out_indices[number_of_features-i-1] > m:
                            left_out_indices.append(m)
                        else:
                            left_out_indices.append(m+1)
                    print(left_out_indices)

            print(number_of_features-i)
        else:
            for k in left_out_indices:
                selected_features[k] = False
            print('break')
            break
else:
    selected_features = np.ones(number_of_features, dtype=bool)
print(selected_features)
print(len(X_train[1]))

#print(error_ridge)

#w_star_lasso = lasso_regression(X_train, y_train, _lambda)
#error_lasso = error_predict(X_val, y_val, w_star_lasso)
#print(w_star_lasso)
#print(error_lasso)
print(w_star_ridge)
w_star = np.zeros(number_of_features)
j=0
for n in range(number_of_features):
    if selected_features[n]==True:
        w_star[n] = w_star_ridge[j]
        j+=1
    else:
        w_star[n] = 0.0
print(w_star)


# Define features
#feature_set = {'Linear1', 'Linear2', 'Linear3', 'Linear4', 'Linear5', 'Squared1', 'Squared2', 'Squared3', 'Squared4', 'Squared5', 'Cos1(x)', 'Cos2(x)', 'Cos3(x)', 'Cos4(x)', 'Cos5(x)', 'Exp1(x)', 'Exp2(x)', 'Exp3(x)', 'Exp4(x)', 'Exp5(x)', 'Constant'}


for i in range(number_of_features):                                             #Write stuff to the submission file
    file.write(str(w_star[i]))
    file.write('\n')
file.close()
