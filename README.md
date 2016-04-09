# CSC411 Assignment 3 README
Authors: Benjamin Lavon and Makram Kamaleddine

Kaggle: benlavon and makram (team name: monty_python)

# Files Submitted:
- svm.py: the python file where we train our SVM classifier
- util.py: the python file with various utility functions to load
the training and testing data, write CSV files, and so on. The training/testing
data must be in the same directory as this file.
- keras_net.py: the python file that creates and trains the KerasNet network
described in the report.
- leaky_arch.py: the python file that creates and trains the KerasNet network with
leaky ReLUs as activations instead of vanilla ReLUs, also discussed in the report.
- convolutional.py: the python file that creates and trains the LeNet network
architecture discussed in the report.
- run_saved_model.py: the python file that runs a saved model with already trained 
weights. 
- test.mat: the mat file with the 1 x K vector of predictions on all the test data
(public and private)

# References 
We used the following libraries:
- TensorFlow (https://www.tensorflow.org/).
- Keras (http://keras.io/). 
- LeNet 5 code found in convolutional.py (https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/mnist/convolutional.py).

