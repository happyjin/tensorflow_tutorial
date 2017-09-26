import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

# declare list of features, we only have one real-valued feature
def model_fn(features, labels, mode):
    # build a linear model and predict values
    # step 1: set variables(in other words, trainable parameters to the graph)
    W = tf.get_variable("W", [1], dtype=tf.float64) # create a new variable named W and it's a real float64 number
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    # step 2: loss sub-graph
    loss = tf.reduce_sum(tf.square(labels - y))
    # step 3: training sub-graph
    # use it to restart training exactly where you left off when the training procedure has been stopped for some reason
    global_step = tf.train.get_global_step() # before starting train the model, we need to set tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # Group all updates to into a single train op.
    # assign global_step with 1
    # assign_add used to simplify the graph creation and will speedup the routine
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # step 4: connect all subgraphs we built using EstimatorSpec
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)

if __name__ == "__main__":
    # define data sets
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])

    # step 1: load numpy training and evaluation data into estimator using numpy_input_fn method
    input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=0, shuffle=True)
    train_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
    eval_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
    # step 2: train the model
    # model_fn that tells tf.estimator how it can evaluate predictions, training steps, and loss
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    estimator.train(input_fn=input_fn, steps=1000)
    # model evaluation
    train_metrics = estimator.evaluate(input_fn=train_fn) # estimate training
    eval_metrics = estimator.evaluate(input_fn=eval_fn)
    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % eval_metrics)