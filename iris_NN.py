from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
import tensorflow as tf
import numpy as np
import urllib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# download datasets
if not os.path.exists(IRIS_TRAINING):
  raw = urllib.urlopen(IRIS_TRAINING_URL).read()
  with open(IRIS_TRAINING,'w') as f:
    f.write(raw)

if not os.path.exists(IRIS_TEST):
  raw = urllib.urlopen(IRIS_TEST_URL).read()
  with open(IRIS_TEST,'w') as f:
    f.write(raw)

# Step 1: load training dataset and test dataset using function in tf
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)

#print training_set.data
#print training_set.target
#print test_set.data
# print test_set.target

### construct a Deep Neural Network Classifier
# step 2: specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])] # there are 4 features in the dataset so that the shape must be 4

# step 3: build 3 layer DNN with 10, 18, 10 units respectively (configure the DNN classifier model)
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, # the set of feature columns defined above
                                        hidden_units=[10, 20, 10], # three hidden layers, containing 10, 20 and 10 respectively
                                        n_classes=3, # three target classes
                                        model_dir="/Users/lijin/Desktop/tmp")
### describe the training input pipeline
# step 4: define the training inputs
# we can use tf.esimator.inputs.numpy_input_fn to produce the input pipeline
train_input_fn = tf.esimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)}, # generate data for the model
    y=np.array(training_set.target), # generate data for the model
    num_epochs=None,
    shuffle=True)

### fit the DNN classifier to the Iris training data
# step 5: train the model (fit it to the Iris training data using the train method)
classifier.train(input_fn=train_input_fn, steps=2000)
# the state of the model is preserved in the classifier, this is equivalent to the following
#classifier.train(input_fn=train_input_fn, steps=1000)
#classifier.train(input_fn=train_input_fn, steps=1000)