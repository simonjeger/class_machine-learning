import numpy as np
from numpy.random import seed
import pandas as pd
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import normalize, to_categorical

### GLOBAL PARAMETERS
activation_seeding = True                                                      #make sure to set this to True while training hyperparams
considered_n_predictions = 20                                                   #sets how many of the first predictions should be kept
activation_image_prediction = False                                             #producing the file 'predictions.csv' (ca. 25mins)
validation_samples = 10000

### PARAMETERS FOR PREDICTION AND DATA CREATION
considered_n_classes = 0                                                        #number of classes where their indices are put to 1.0 in the trainig data
considered_n_probabilities = 20
activation_train_data = False                                                   #train and test matrix are being built if set to TRUE
activation_test_data = False

### FOR CHOOSING THE RIGHT HYPERPARAMS
hypertraining_iterations = 20
activation_hypertuning = True
if activation_hypertuning==True:
    hyperarr_epochs = [25]                                                          #old sets: [1, 5, 10, 20, 30]
    hyperarr_batchsize = [32]                                                       #old sets: [20, 32, 32, 50, 75]
    hyperarr_n_layers = [2, 3, 4, 5]                                                      #old sets: [1, 2, 3, 4]
    hyperarr_start_density = [100, 300, 500]                                             #old sets: [100, 300, 500]
    hyperarr_dropout = [0.1, 0.2, 0.3]                                              #old sets: [0.1, 0.3, 0.5]
    hyperarr_choosing_label_threshold = [0.5]                                        #old sets: [0.1, 0.3, 0.5]
else:
    hyperarr_epochs = [5]
    hyperarr_batchsize = [32]
    hyperarr_n_layers = [3]
    hyperarr_start_density = [500]
    hyperarr_dropout = [0.2]
    hyperarr_choosing_label_threshold = [0.1]

activation_hack_train_data = False                                              #to include most certain predictions as new training data (approx. 30'000/59'500 triplets)
unseeded_submission_iterations = 1                                              #creating different submissions files for 'the_hack.py' --> set 'activation_seeding' to False

def predict_image_data(number_pictures):
    model = VGG16()
    M_prediction_classes = []
    M_prediction_probabilities = []
    index_image = []
    for i in range(number_pictures):
        image_number = str(i)
        image_number = image_number.zfill(5)
        index_image.append(image_number)                                        #append number of image name for prediction file

        image = load_img('../data_4/food/' + image_number + '.jpg', target_size=(224, 224))  #min number = 00000, max number = 09999
        print(image_number, '.jpg')
        image = img_to_array(image)                                             #convert the image pixels to a numpy array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))#reshape data for the model
        image = preprocess_input(image)                                         #prepare the image for the VGG model
        yhat = model.predict(image)                                             #predict the probability across all output classes
        yhat = decode_predictions(yhat, top=1000)                               #convert the probabilities to class labels
        prediction_matrix = pd.DataFrame(data=yhat[0][:])                      #retrieve the most likely result, e.g. highest probability
        #print(probability_matrix)
        image_classes = []
        for row in prediction_matrix[0][0:considered_n_predictions]:
            numbers = row.split('n')
            image_classes.append(int(numbers[1]))
        image_probabilities = prediction_matrix[2][0:considered_n_predictions]
        M_prediction_classes.append(image_classes)
        M_prediction_probabilities.append(image_probabilities)
    M_prediction = np.c_[index_image, M_prediction_classes, M_prediction_probabilities]
    M_prediction_pd = pd.DataFrame(data=M_prediction, columns=["image_name", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9", "class_10", "class_11", "clas_12", "class_13", "class_14", "class_15", "class_16", "class_17", "class_18", "class_19", "class_20", "probability_1", "probability_2", "probability_3", "probability_4", "probability_5", "probability_6", "probability_7", "probability_8", "probability_9", "probability_10", "probability_11", "probability_12", "probability_13", "probability_14", "probability_15", "probability_16", "probability_17", "probability_18", "probability_19", "probability_20"])
    print(M_prediction_pd)
    M_prediction_pd.to_csv(r'predictions_' + str(considered_n_predictions) + '.csv', index=False, float_format='%.5f')

    return

def read_class_data():
    ### LOAD PREDICTIONS
    predictions = pd.read_csv('predictions_20.csv', delimiter=',')
    image_number = predictions.iloc[:,0]
    image_number = image_number.map(lambda x: f'{x:0>5}')                       #append zeros in the front
    image_class = predictions.iloc[:,1:(considered_n_predictions+1)]
    image_probabilities = predictions.iloc[:,(considered_n_predictions+1):(2*considered_n_predictions+1)]

    ### LOAD TRAIN DATA
    train_triplets_plain = pd.read_csv('../data_4/train_triplets.txt', delimiter=' ', header=None)
    if activation_hack_train_data:
        hack_train_triplets = pd.read_csv('../data_4/hack_train_triplets.txt', delimiter=',', header=None)
        train_triplets = pd.concat([train_triplets_plain, hack_train_triplets]).reset_index(drop=True)
    else:
        train_triplets = train_triplets_plain

    ### LOAD TEST DATA
    test_triplets = pd.read_csv('../data_4/test_triplets.txt', delimiter=' ', header=None)     #initial test set for uploading submissions

    return image_number, image_class, image_probabilities, train_triplets, test_triplets

def create_class_array(image_class):
    class_array = []
    for i in range(len(image_class)):
        for j in range(considered_n_classes+considered_n_probabilities):
            if class_array.count(image_class.iloc[i,j])==0:
                class_array.append(image_class.iloc[i,j])

    return class_array, len(class_array)

def create_data_point(triplets, i, image_class, image_probabilities, class_array):
    #FOR IMAGE IN COLUMN 'A': FILL MATRIX WITH 1 FOR 'considered_n_classes' AND PROBABILITY FOR 'considered_n_probabilities'
    X = np.zeros(3*len(class_array), dtype=int)
    classes_first_image = image_class.iloc[triplets[0][i],0:considered_n_classes]                               #take first 'considered_n_classes' class numbers and put them to classes_first_image
    classes_first_image = np.asarray(classes_first_image)
    probability_first_image = image_class.iloc[triplets[0][i],considered_n_classes:(considered_n_classes+considered_n_probabilities)]  #take the following 'considered_n_probabilities' class numbers and put them to probability_first_image
    probability_first_image = np.asarray(probability_first_image)
    for j_class in range(considered_n_classes):
        index_class_first_image = class_array.index(classes_first_image[j_class])
        X[index_class_first_image] = 1
    for j_proba in range(considered_n_probabilities):
        index_proba_first_image = class_array.index(probability_first_image[j_proba])
        X = np.array(X, dtype=float)
        X[index_proba_first_image] = image_probabilities.iloc[triplets[0][i], j_proba+considered_n_classes]

    #FOR IMAGE IN COLUMN 'B': FILL MATRIX WITH 1 FOR 'considered_n_classes' AND PROBABILITY FOR 'considered_n_probabilities'
    classes_second_image = image_class.iloc[triplets[1][i],0:considered_n_classes]
    classes_second_image = np.asarray(classes_second_image)
    probability_second_image = image_class.iloc[triplets[1][i],considered_n_classes:(considered_n_classes+considered_n_probabilities)]
    probability_second_image = np.asarray(probability_second_image)
    for k_class in range(considered_n_classes):
        index_class_second_image = class_array.index(classes_second_image[k_class])
        X[index_class_second_image+len(class_array)] = 1
    for k_proba in range(considered_n_probabilities):
        index_proba_second_image = class_array.index(probability_second_image[k_proba])
        X = np.array(X, dtype=float)
        X[index_proba_second_image+len(class_array)] = image_probabilities.iloc[triplets[1][i], k_proba+considered_n_classes]

    #FOR IMAGE IN COLUMN 'C': FILL MATRIX WITH 1 FOR 'considered_n_classes' AND PROBABILITY FOR 'considered_n_probabilities'
    classes_third_image = image_class.iloc[triplets[2][i],0:considered_n_classes]
    classes_third_image = np.asarray(classes_third_image)
    probability_third_image = image_class.iloc[triplets[2][i],considered_n_classes:(considered_n_classes+considered_n_probabilities)]
    probability_third_image = np.asarray(probability_third_image)
    for l_class in range(considered_n_classes):
        index_class_third_image = class_array.index(classes_third_image[l_class])
        X[index_class_third_image+2*len(class_array)] = 1
    for l_proba in range(considered_n_probabilities):
        index_proba_third_image = class_array.index(probability_third_image[l_proba])
        X = np.array(X, dtype=float)
        X[index_proba_third_image+2*len(class_array)] = image_probabilities.iloc[triplets[2][i], l_proba+considered_n_classes]

    return X

def create_train_data(train_triplets, image_class, image_probabilities, class_array):
    X_TRAIN = []
    Y_LABELS = []
    ### CREATING ENTRIES WITH POSITIVE LABELS USING FIRST HALF OF TRAINING DATA
    for i in range(len(train_triplets)):
        print('train true:', i)
        X = create_data_point(train_triplets, i, image_class, image_probabilities, class_array)
        X_TRAIN.append(X)
        Y_LABELS.append(1)

    ### SWAP THE COLUMNS
    triplets_temp = np.asarray(train_triplets)
    triplets_swaped = np.c_[triplets_temp[:,0], triplets_temp[:,2], triplets_temp[:,1]]
    train_false_triplets = pd.DataFrame(data=triplets_swaped)

    ### CREATING ENTRIES WITH NEGATIVE LABELS USING SECOND HALF OF TRAINING DATA
    for j in range(len(train_false_triplets)):
        print('train false:', j)
        X = create_data_point(train_false_triplets, j, image_class, image_probabilities, class_array)
        X_TRAIN.append(X)
        Y_LABELS.append(0)

    print('Writing train and test matrix to csv file...')
    M_TRAIN_pd = pd.DataFrame(data=X_TRAIN)
    M_LABELS_pd = pd.DataFrame(data=Y_LABELS)
    M_TRAIN_pd.to_csv(r'train_matrix.csv', index=False)
    M_LABELS_pd.to_csv(r'train_labels.csv', index=False)
    return

def create_test_data(test_triplets, image_class, image_probabilities, class_array):
    X_TEST = []
    for i in range(len(test_triplets)):
        print('test', i)
        X = create_data_point(test_triplets, i, image_class, image_probabilities, class_array)
        X_TEST.append(X)

    M_TEST_pd = pd.DataFrame(data=X_TEST)
    M_TEST_pd.to_csv(r'test_matrix.csv', index=False)
    return

def read_model_data():
    X_TRAIN_readin = pd.read_csv('train_matrix.csv', delimiter=',')
    Y_LABELS_readin = pd.read_csv('train_labels.csv', delimiter=',')
    Y_LABELS_readin = Y_LABELS_readin.iloc[:,0]
    X_TEST = pd.read_csv('test_matrix.csv', delimiter=',')

    X_TRAIN, X_VALIDATION, Y_LABELS, Y_VALIDATION = train_test_split(X_TRAIN_readin, Y_LABELS_readin, test_size = 0.2, random_state = 42)

    return np.asarray(X_TRAIN), np.asarray(X_VALIDATION), np.asarray(Y_LABELS), np.asarray(Y_VALIDATION), np.asarray(X_TEST)

def biene_maya_choosing(y_probability, hyperparam_choosing_label_threshold):
    y_pred = []
    for i in range(len(y_probability)):
        if y_probability[i] > hyperparam_choosing_label_threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

#------------------------------  MAIN  -----------------------------------------
### SEEDING THE NUMPY RANDOM GENERATOR AND THE TENSORFLOW RANDOM GENERATOR
if activation_seeding == True:
    #seed(1)
    tf.random.set_seed(3)

### PREDICT THE FIRST 'CONSIDERED_N_PREDICTIONS' CLASSES AND THE CORRESPONDING PROBABILITIES
if activation_image_prediction:
    number_pictures = 10000
    print('Predicting image classes and image probabilities...')
    predict_image_data(number_pictures)

### PREPARE FOR GENERATING TRAIN AND TEST DATA
if activation_train_data or activation_test_data:
    #read in the class data from the predictions.csv
    print('Reading class data...')
    [image_number, image_class, image_probabilities, train_triplets, test_triplets] = read_class_data()

    #create an array that contains every predicted class only once
    print('Creating class array...')
    [class_array, n_different_labels] = create_class_array(image_class)
### BUILDING UP THE WHOLE TRAIN DATA
if activation_train_data:
    #CREATE TWO DATA POINTS FOR EACH ROW IN 'train_triplets.csv'
    print('Preprocessing training data...')
    create_train_data(train_triplets, image_class, image_probabilities, class_array)
### BUILDING UP THE WHOLE TEST DATA
if activation_test_data:
    #CREATE TWO DATA POINTS FOR EACH ROW IN 'test_triplets.csv'
    print('Preprocessing testing data...')
    create_test_data(test_triplets, image_class, image_probabilities, class_array)

### READ IN DATA FOR MODEL TRAINING
[X_TRAIN, X_VALIDATION, Y_LABELS, Y_VALIDATION, X_TEST] = read_model_data()

### SETUP NEURAL NETWORK FOR TUNING THE HYPERPARAMS
tuning_params = []
y_probability = []
for n in range(hypertraining_iterations):
    print('Building up structure of NN...')

    ### SETTING RANDOM HYPERPARAMETERS
    hyperparam_epochs = hyperarr_epochs[np.random.randint(0,len(hyperarr_epochs))]
    hyperparam_batchsize = hyperarr_batchsize[np.random.randint(0,len(hyperarr_batchsize))]
    hyperparam_n_layers = hyperarr_n_layers[np.random.randint(0,len(hyperarr_n_layers))]
    hyperparam_start_density = hyperarr_start_density[np.random.randint(0,len(hyperarr_start_density))]
    hyperparam_dropout = hyperarr_dropout[np.random.randint(0,len(hyperarr_dropout))]
    hyperparam_choosing_label_threshold = hyperarr_choosing_label_threshold[np.random.randint(0,len(hyperarr_choosing_label_threshold))]

    ### BUILDING NEURAL NETWORK ACCORDING TO HYPERPARAMETERS
    prediction_model = tf.keras.models.Sequential()
    #prediction_model.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN[0]),)))
    for n_layers in range(hyperparam_n_layers):
        prediction_model.add(tf.keras.layers.Dense(int(hyperparam_start_density/np.power(2,n_layers)), activation='relu'))
        prediction_model.add(tf.keras.layers.Dropout(hyperparam_dropout))
    prediction_model.add(tf.keras.layers.Dense(1, activation='relu'))
    prediction_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    mode_path = 'best_model.h5'
    callbacks = [ModelCheckpoint(filepath=mode_path, save_best_only=True, monitor='val_accuracy', mode='max')]

    ### TRAINING THE BUILT MODEL
    print('Started training neural network...')
    prediction_model.fit(X_TRAIN, Y_LABELS, validation_data=(X_VALIDATION, Y_VALIDATION), batch_size=hyperparam_batchsize, epochs=hyperparam_epochs, callbacks=callbacks, verbose=1)
    final_model = load_model('best_model.h5')
    y_pred_validation = final_model.predict_classes(X_VALIDATION)           #formerly y_probability_validation
    y_pred_test = final_model.predict_classes(X_TEST)                       #formerly y_probability_test
    del prediction_model

    ### CHOOSE LABEL ACCORDING TO FIRST OR SECOND PROBABILITY BEING HIGHER
    #y_pred_validation = biene_maya_choosing(y_probability_validation, hyperparam_choosing_label_threshold)
    #y_pred_test = biene_maya_choosing(y_probability_test, hyperparam_choosing_label_threshold)

    count_accuracy = 0
    for i_acc in range(len(y_pred_validation)):
        if y_pred_validation[i_acc]==Y_VALIDATION[i_acc]:
            count_accuracy += 1

    ### DEFINE ACCURACY OF BOTH CERTAIN TRUE AND CERTAIN FALSE LABELS AND APPEND THE APPLIED HYPERPARAMETERS TO SETTINGS
    #print('accuracy is: ', count_accuracy, 'out of', len(y_pred))
    tuning_params.append([hyperparam_epochs, hyperparam_batchsize, hyperparam_n_layers, hyperparam_start_density, hyperparam_dropout, hyperparam_choosing_label_threshold, count_accuracy])
    print(tuning_params)

    ### WRITING SOLUTION TO SUBMISSION .CSV-FILE
    M_submission_pd = pd.DataFrame(data=y_pred_test)
    M_submission_pd.to_csv(r'submission_files/submission_' + str(n) + '.csv', index=False, header=False)

### WRITING PARAMETERS TO FILE 'hyperparam_settings.csv'
M_hyperparam_settings = pd.DataFrame(data=tuning_params, columns=["epochs", "batch_size", "n_layers_NN", "start_density", "dropout", "choosing_label_threshold", "accuracy"])
print(M_hyperparam_settings)
M_hyperparam_settings.to_csv(r'hyperparam_settings.csv')
