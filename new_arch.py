from __future__ import print_function
import numpy as np
import sys

if len(sys.argv) != 4:
    raise ValueError("Need csv filename, model json filename, model weights filename")

csv_filename = sys.argv[1]
json_filename = sys.argv[2]
weights_filename = sys.argv[3]

print('Saving predictions to: ', csv_filename)
print('Saving model to: ', json_filename)
print('Saving weights to: ', weights_filename)

np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils

from util import *

batch_size = 16
nb_classes = 7
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 32, 32
# number of conv filters
nb_filters = 32
high_filters = 64
# size of pooling
nb_pool = 2
# conv kernel size
nb_conv = 3
# fc layer size
fc_size = 128

# the data, shuffled and split between train and test
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
train, test = load_data()

X_train = np.transpose(train['tr_images'])
y_train = train['tr_labels']
y_train = y_train.reshape((len(y_train), ))
y_train = y_train - np.ones((len(y_train), ))
X_test = np.transpose(test['public_test_images'])

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

# add zero padding to keep size the same after first conv layer
# alexnet architecture:
# conv-conv-pool-conv-pool-fc-fc-softmax

# zero padding to keep size constant after first conv
model.add(ZeroPadding2D(padding=(1, 1), input_shape=(1, img_rows, img_cols)))

# first conv layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
    border_mode='valid'))
model.add(Activation('relu'))

# second conv layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))

# max pool layer
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

# 3rd conv
model.add(Convolution2D(nb_filters, nb_conv - 1, nb_conv - 1))
model.add(Activation('relu'))

# max pooling
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

# fc layers
model.add(Flatten())
model.add(Dense(fc_size))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        show_accuracy=True, verbose=1)
score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=0)

# get predicted scores
predicted = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
predicted = predicted + np.ones((len(predicted), ))
predicted = np.asarray(predicted, dtype='int32')

# write predictions to csv file
write_csv(csv_filename, predicted)

# save model to json file
open(json_filename, 'w').write(model.to_json())

# save weights to h5 file and mat file
model.save_weights(weights_filename)
model.save_weights(weights_filename + '.mat')

print('Test score: ', score[0])
print('Test accuracy: ', score[1])
