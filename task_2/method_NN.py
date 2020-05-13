import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import normalize, to_categorical

def neural_network(model_number, X_TRAIN, Y_LABELS, X_TEST, n_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN[0]),)))
    model.add(tf.keras.layers.Dense(190, activation='sigmoid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(95, activation='sigmoid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_TRAIN, Y_LABELS, batch_size=32, epochs=15, verbose=1)
    if model_number == 1:
        y = model.predict(X_TEST)
    elif model_number == 2:
        y = model.predict_classes(X_TEST)
    return y
