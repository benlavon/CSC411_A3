import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, metrics

# csv writer
from util import *

# hyperparameters
GAMMA = 0.000001
# kernel options: linear, poly, rbf, sigmoid, precomputed
KERNEL = 'poly'
# degree of the polynomial kernal function
DEGREE = 2
# tolerance
TOL = 1e-9
# penalty parameter of error term
C = 100.0

training_data = scipy.io.loadmat('labeled_images.mat')
testing_data  = scipy.io.loadmat('public_test_images.mat')

# training data
identity = training_data['tr_identity']
images = np.transpose(training_data['tr_images'])
labels = training_data['tr_labels']

# testing data
test_images = np.transpose(testing_data['public_test_images'])
test_data = test_images.reshape((len(test_images), -1))

n_samples = len(images)
data = images.reshape((n_samples, -1))

# create svm classifier
classifier = svm.SVC(gamma=GAMMA, C=C, kernel=KERNEL, degree=DEGREE, tol=TOL)

# train the classifier
classifier.fit(data[:n_samples / 2], labels[:n_samples/2])

# predict some values
expected = labels[n_samples/2:]
predicted = classifier.predict(data[n_samples/2:])

predicted2 = classifier.predict(test_data)
# write to csv file
write_csv('predictions_svm.csv', predicted2)

print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted))
