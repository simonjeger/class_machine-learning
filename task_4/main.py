import csv
import sys
import requests
import skimage.io
import os
import glob
import pickle
import time

from IPython.display import display, Image, HTML
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image as kimage
from keras.models import load_model
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skimage.io

sys.path.append('../')
import helpers

rand_img = np.random.choice(glob.glob('../data_4/food/*.jpg'))
Image(filename=rand_img)
img = skimage.io.imread(rand_img)
#print(img.shape)

#RESIZE IMAGE TO FIT PRETRAINED MODEL
img = kimage.load_img(rand_img, target_size=(224, 224))
x = kimage.img_to_array(img)
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
#x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#print(x.shape)

#image_top=False removes final connected layers
"""if 1:
    model = VGG16(include_top=False, weights='imagenet')
    model.save('VGG16.h5')
else:
    load_model('VGG16.h5')"""
model = VGG16(include_top=False, weights='imagenet')
pred = model.predict(x)
print(pred)
print(len(pred))
print(len(pred[0]))
print(len(pred[0][0]))
print(len(pred[0][0][0]))

label = decode_predictions(pred)
label = label[0][0]
#print(pred.shape)
#print(pred.ravel().shape)



"""#BLACK MAGIC
df = pd.read_csv('../data/model_likes_anon.psv',
                 sep='|', quoting=csv.QUOTE_MINIMAL,
                 quotechar='\\')
df.drop_duplicates(inplace=True)
df = helpers.threshold_interactions_df(df, 'uid', 'mid', 5, 5)

# model_ids to keep
valid_mids = set(df.mid.unique())"""
