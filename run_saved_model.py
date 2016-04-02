from keras.models import model_from_json
from util import *
import sys

if len(sys.argv) != 4:
    print_usage()
    sys.exit()

json_filename = sys.argv[1]
h5_filename = sys.argv[2]

model = model_from_json(open(json_filename).read())
model.load_weights(h5_filename)

# get training and test data
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

# get score on the training set
score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=0)

# get predicted scores on the test set
predicted = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
predicted = predicted + np.ones((len(predicted), ))
predicted = np.asarray(predicted, dtype='int32')

# write predictions to csv file
write_csv(csv_filename, predicted)
