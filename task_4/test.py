import numpy as np
import pandas as pd
import yaml                                                                     #for reading running parameters from external yaml file (Euler)
import argparse

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

from sklearn import svm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import normalize, to_categorical

considered_n_predictions = 20
activation_image_prediction = False
considered_n_classes = 0
considered_n_probabilities = 20
activation_train_test_data = False
model_type = 'neural'                                                           #either 'neural' or 'svm'
activation_hack_train_data = False

def create_test_data(test_triplets, image_class, image_probabilities, class_array):
    X_TEST = []
    for i in range(len(test_triplets)):
    #for i in range(1):                                                          #for debugging purposes
        #print('test', i)                                                        #progress bar if the waiting takes too long:)
        X = create_data_point(test_triplets, i, 'B', image_class, image_probabilities, class_array)
        X_TEST.append(X)

    M_TEST_pd = pd.DataFrame(data=X_TEST)
    M_TEST_pd.to_csv(r'test_from_train_matrix.csv', index=False)
    return
