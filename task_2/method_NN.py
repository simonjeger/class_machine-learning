import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import normalize, to_categorical

def neural_network_1(X_TRAIN, Y_LABELS, X_TEST, hyperparam_epochs, hyperparam_batchsize, hyperparam_n_layer, hyperparam_start_density, hyperparam_dropout):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN[0]),)))
    for n_layers in range(hyperparam_n_layers):
        prediction_model.add(tf.keras.layers.Dense(int(hyperparam_start_density/np.power(2,n_layers)), activation='sigmoid'))
        prediction_model.add(tf.keras.layers.Dropout(hyperparam_dropout))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(len(Y_LABELS[0]), activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_TRAIN, Y_LABELS, batch_size=hyperparam_batchsize, epochs=hyperparam_epochs, verbose=1)
    y = model.predict(X_TEST)
    return y

def neural_network_2(X_TRAIN, Y_LABELS, X_TEST):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN[0]),)))
    model.add(tf.keras.layers.Dense(400, activation='sigmoid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(200, activation='sigmoid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_TRAIN, Y_LABELS, batch_size=32, epochs=15, verbose=1)

    y = model.predict_classes(X_TEST)
    return y


def neural_network_3(X_TRAIN, Y_LABELS, X_TEST):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(len(X_TRAIN[0]),)))
    model.add(tf.keras.layers.Dense(190, activation='sigmoid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(95, activation='sigmoid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(Y_LABELS[0]), activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_TRAIN, Y_LABELS, batch_size=32, epochs=15, verbose=1)

    y = model.predict_classes(X_TEST)
    return y
