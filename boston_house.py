from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import pandas as pd
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# step 1: set logging verbosity to INFO for more detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# step 2: import the housing data
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"
training_set = pd.read_csv("datasets/boston_train.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv("datasets/boston_test.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("datasets/boston_predict.csv", skipinitialspace=True, skiprows=1, names=COLUMNS)

# step 3: create a list of FeatureColumn for the input data which specify the set of features to use for training
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# step 4: instantiate a DNNRegressor for the neural network regression model
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10],
                                      model_dir="/Users/lijin/Desktop/tmp")

# step 5: build the input_fn. pass input data into regressor
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x = pd.DataFrame({k: data_set[k].values for k in FEATURES}), # pass any data_set include training_set, test_set
        y = pd.Series(data_set[LABEL].values),
        # num_epochs controls the number of epochs to iterate over data. For training, set this to None
        # input_fn will iterate over once and then raise OutOfRangeError
        num_epochs=num_epochs,
        # For evaluate and predict, set this to False so that iterate over the data sequentially
        # For train, set this to True
        shuffle=shuffle)

# step 6: train the neural network regressor
regressor.train(input_fn=get_input_fn(training_set), steps=5000)

# step 7: evaluate the model
ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))

# step 8: retrieve the loss from the evaluation results and print it to output
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

# step 9: