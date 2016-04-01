import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, metrics

# hyperparameters
GAMMA = 0.001
# kernel options: linear, poly, rbf, sigmoid, precomputed
KERNEL = 'poly'
# degree of the polynomial kernal function
DEGREE = 3
# tolerance
TOL = 1e-5
# penalty parameter of error term
C = 1.0

training_data = scipy.io.loadmat('labeled_images.mat')
testing_data  = scipy.io.loadmat('public_test_images.mat')

# training data
identity = training_data['tr_identity']
images = np.transpose(training_data['tr_images'])
labels = training_data['tr_labels']

# testing data
test_images = np.transpose(testing_data['public_test_images'])
test_data = test_images.reshape((len(test_images), -1))

images_and_labels = list(zip(images, labels))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(images)
data = images.reshape((n_samples, -1))

# create svm classifier
classifier = svm.SVC(gamma=GAMMA, C=C, kernel=KERNEL, degree=DEGREE, tol=TOL)

# train the classifier
classifier.fit(data[:n_samples / 2], labels[:n_samples/2])

# predict some values
expected = labels[n_samples/2:]
predicted = classifier.predict(data[n_samples/2:])

print "Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted))

images_and_predictions = list(zip(images[n_samples/2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
