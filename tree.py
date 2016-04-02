from sklearn import tree
from util import *

import numpy as np

training_data, testing_data = load_data()

images = np.transpose(training_data['tr_images'])
labels = training_data['tr_labels']

test_images = np.transpose(testing_data['public_test_images'])
test_data = test_images.reshape((len(test_images), -1))

n_samples = len(images)
data = images.reshape((n_samples, -1))

# create the dtree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, labels)

predicted = clf.predict(test_data)

write_csv('predictions_tree.csv', predicted)
