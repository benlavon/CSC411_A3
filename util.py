import scipy.io
import numpy as np

def write_csv(filename, predictions):
    """
    Write the given predictions to a csv file.
    :param filename: the name of the csv file
    :param predictions: the predictions to write
    """
    with open(filename, 'w') as f:
        f.write('Id,Prediction\n')
        lines_so_far = 1
        for index, prediction in enumerate(predictions):
            f.write(str(index + 1))
            f.write(',')
            f.write(str(prediction))
            f.write('\n')
            lines_so_far += 1

        print lines_so_far
        while lines_so_far <= 1253:
            f.write(str(lines_so_far))
            f.write(',')
            f.write(str(0))
            f.write('\n')
            lines_so_far += 1


def load_data():
    """
    Load the labeled training and testing data.
    :return: two dictionaries, the first one containing training data arrays, and the
    second one containing the test data set.
    """
    training_data = scipy.io.loadmat('labeled_images.mat')
    testing_data  = scipy.io.loadmat('public_test_images.mat')

    return training_data, testing_data

def to_categorical(y, nb_classes):
    y_ = np.asarray(y, dtype='int32')
    Y = np.zeros((len(y_), nb_classes))
    for i in range(len(y_)):
        Y[i, y_[i] -1] = 1.
    return Y

