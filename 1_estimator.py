# tf.estimator is a high-level TensorFlow library that simplifies the mechanics of machine learning,
# including the following: 1. running training loops 2. running evaluation loops 3. managing data sets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
print tf.__version__
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# step 1: declare list of features(in other words, input data).
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])] # we only have one numeric feature
# step 2: declare type of estimator(LinearRegressor in this case) to train and evaluate
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns) # use LinearRegressor for input data x
# step 3: define training and evaluation data using numpy format data
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])
# step 4: load numpy training and evaluation data into estimator using numpy_input_fn method
input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True) # input specific real data for training
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False) # input data with epochs for evaluation of train
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False) # input data with epochs for evaluation of test
# step 5: invoke 1000 training steps using train method of estimator
estimator.train(input_fn=input_fn, steps=1000) # input x_train and y_train for training in the estimator
# step 6: evaluate how well the model did
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
# step 7: print the results
print("train metrics: %r" % train_metrics) # training result information
print("eval metrics: %r" % eval_metrics) # test(evaluation) result information