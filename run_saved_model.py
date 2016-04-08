from keras.models import model_from_json
from keras.utils import np_utils
from util import *
import sys

if len(sys.argv) != 4:
    print_usage()
    sys.exit()

json_filename = sys.argv[1]
h5_filename = sys.argv[2]
csv_filename = sys.argv[3]

model = model_from_json(open(json_filename).read())
model.load_weights(h5_filename)

# get training and test data
test = load_hidden()
training, public_test = load_data()

# number of classes
nb_classes = 7
# batch size
batch_size = 32
# input image dimensions
img_rows, img_cols = 32, 32
# number of conv filters
nb_filters = 32
# size of pooling
nb_pool = 2
# conv kernel size
nb_conv = 3

X_test = np.transpose(test['hidden_test_images'])
X_public = np.transpose(public_test['public_test_images'])

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_public = X_public.reshape(X_public.shape[0], 1, img_rows, img_cols)
X_test = X_test.astype('float32')
X_public = X_public.astype('float32')
X_test /= 255
X_public /= 255
X_all = np.concatenate((X_public, X_test))
print(X_public.shape[0], 'public test samples')
print(X_test.shape[0], 'hidden test samples')
print(X_all.shape[0], 'total images')

# get predicted scores on the test set
predicted = model.predict_classes(X_all, batch_size=batch_size, verbose=1)
predicted = predicted + np.ones((len(predicted), ))
predicted = np.asarray(predicted, dtype='int32')

# write predictions to csv file
write_csv_proper(csv_filename, predicted)
