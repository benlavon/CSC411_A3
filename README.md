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
- savedModel.h5: the mat file with the weights of the trained KerasNet network. This is an
h5 file which can be loaded into MATLAB using one of the `h5*` functions (see here: http://www.mathworks.com/help/matlab/import_export/importing-hierarchical-data-format-hdf5-files.html)
or in Python using the h5py library (here: http://www.h5py.org/).
- model.json : the JSON file that holds information about the neural network architecture used
for KerasNet. 

# References 
We used the following libraries and code:
- TensorFlow (https://www.tensorflow.org/).
- Keras (http://keras.io/). 
- SciKit-Learn (http://scikit-learn.org/).
- Modified LeNet 5 code found in convolutional.py (https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/mnist/convolutional.py).
- Used the following file as a basis for Keras files: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
- Used the following page as a basis for `svm.py`: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

# References to papers used:
- Paper by Hinton et al. explaining dropout: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
- Paper by Krizhevsky et al. presenting AlexNet: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
- Paper by LeCun presenting LeNet and other findings: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

# Running the Network
In order to run the given network, with the given `model.json` file and `savedModel.h5` file,
you will have to install Tensorflow (see link above), Keras (see link above), configure Keras with
TensorFlow rather than Theano, and use the `run_saved_model.py` python file to classify all the
public and private test data with the network. This file will produce as output a CSV file with
1253 classifications, the first 418 being the public test set, and the rest being the private test
set.

`run_saved_model.py` is run like this:
```
python run_saved_model.py path_to_model_json path_to_weights_h5 path_to_csv_output
```

If you extract all the files into the same directory you can do this:
```
python run_saved_model.py model.json savedModel.h5 full_output.csv
```

And this will produce a CSV file called `full_output.csv` with predictions for all the test
data. It will also produce a file `full_output.csv.mat` which is a 1 by 1253 matrix of all
the predictions that can be opened in MATLAB via `load` or in python via SciPy's `loadmat`
function. 
