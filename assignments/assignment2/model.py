from settings import *
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
import keras.backend as K
from itertools import product
import numpy as np
from keras import regularizers

from keras.utils import plot_model

def w_categorical_crossentropy(y_true, y_pred):
    weights = np.array([[(i - j) ** 2 for i in range(5)] for j in range(5)])
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), 'float32')
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def initialize_model():

    model = Sequential()
    model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='valid',
                     input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])

    print (model.summary())

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

