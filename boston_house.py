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

# step 4: