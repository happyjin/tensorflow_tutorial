import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# set information on TensorFlow logging
tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """
    model function for CNN
    :param features: This is the first item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `Tensor` or `dict` of same.
    :param labels: This is the second item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `Tensor` or `dict` of same (for multi-head models). If
                 mode is `ModeKeys.PREDICT`, `labels=None` will be passed. If
                 the `model_fn`'s signature does not accept `mode`, the
                 `model_fn` must still be able to handle `labels=None`.
    :param mode: Optional. Specifies if this training, evaluation or
                 prediction. See `ModeKeys`
    :return: `EstimatorSpec`
    """""
    # step 1: input layer (tf.layer will accepts a tensor as input) so that we need to
    # reshape features into tensorflow data format(tensor) using tf.reshape
    # and tf.layers always input and output a tensor
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # step 2: convolutional layer #1 and pooling layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2) # [2, 2]=[pool_height, pool_width]

    # step 5: convolutional layer #1 and pooling layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # step 6: dense layer, in other words fully connected layer (tf.layer output a tensor)
    # step 6.1: flat 2D feature maps into 1D in order to utilize in the dense(FC) layer
    # 7*7*64 is computed and means there will be 64*[7,7] feature maps
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    # step 6.2: dense(FC) layer, units=1024 means hidden layer in this FC layer has 1024 neurons
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # step 6.3: set dropout strategy for tf.layers.dense (FC) layer during training time
    # "mode == tf.estimator.ModeKeys.TRAIN" to judge if it is training time or not. dropout only
    # happens in training time, not valid for test time
    dropout = tf.layers.dropout(dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # step 7: logits layer
    # step 7.1: tf.layers.dense output tensor
    # there are 10 cases for the output
    logits = tf.layers.dense(inputs=dropout, units=10)
    # step 7.2: parameter setting for predictions. set "classes" and its "probability"
    predictions = {
        # step 7.2.1: generate predictions (for PREDICT and EVAL mode)
        # we want to find the largest value along the dimension with index of 1, which corresponds to our predictions
        # (recall that our logits tensor has shape [batch_size, 10])
        "classes": tf.argmax(input=logits, axis=1),
        # step 7.2.2: add and named 'softmax_tensor' to the graph. it is used for PREDICT and by the 'logging_hook'. softmax will
        # produce a probability for each cases in 10 units of this case
        "probability": tf.nn.softmax(logits, name="soft_tensor")
    }
    # step 7.3: if this is for prediction then return or print the prediction result
    if mode == tf.estimator.ModeKeys.PREDICT:
        # `EstimatorSpec` is fully defines the model to be run by `Estimator`.
        # 'predictions': `Tensor` or dict of `Tensor`
        # mode: A `ModeKeys`. Specifies if this is training, evaluation or prediction
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # step 8: calculate the loss (for both TRAIN and EVAL mode)
    # step 8.1: generate label into one-hot vector
    # A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    # step 8.2: compute the loss using softmax_cross_entropy in the tf.losses.softmax_cross_entropy
    # compare onehot_labels(ground truth) and logits
    # after computing and get loss which is a scalar Tensor
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # step 9: configure the training op. (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # step 9.1: choose training optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # step 9.2: using optimizer to minimize the `loss` by updating `var_list`
        # global_step just keeps track of the number of batches seen so far.
        # global step record the total number of iterations, maybe used for
        # changing learning rate or other hyperparameter.
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        # step 9.3: return the result
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # step 10: add evaluation matrices
    # step 10.1: define eval_metric_ops dict in EVAL mode as follows
    eval_metric_ops = {
        # tf.metrics.accuracy: calculates how often `predictions` matches `labels`. and return a `Tensor` representing the accuracy
        # predictions: The predicted values, a `Tensor` of any shape.
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    # step 10.2: return the evaluation result
    # eval_metric_ops: Dict of metric results keyed by name. The values of
    # the dict are the results of calling a metric function
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    """
    load the training and test data
    :param unused_argv:
    :return:
    """
    # step 1: load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # return np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # step 2: create the estimator
    # Constructs an `Estimator` instance. cnn_model_fn is `EstimatorSpec`
    # which is fully defines the model to be run by `Estimator`.
    # model_fn argument specifies the model function to use for training, evaluation, and prediction
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/Users/lijin/Desktop/tmp")

    # step 3: set up a logging hook in order to track progress during training
    # step 3.1: define dictionary
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # step 3.2: set up logging
    # since cnn can take a while to train, let's set up some logging so we can track progress during training
    # Prints the given tensors every N local steps, every N seconds, or at end.
    # tensors: store a dict of the tensors we want to log in tensors_to_log, Each key is a label of our choice
    # tensors: `dict` that maps string-valued tags to tensors/tensor names, our probabilities can be found in softmax_tensor
    # probabilities should be logged after every 50 steps of training.
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # step 4: train the model
    # step 4.1: set up the train input function
    # Input function returning a tuple of:
    # features - `Tensor` or dictionary of string feature name to `Tensor`.
    # labels - `Tensor` or dictionary of `Tensor` with labels.
    # batch_size=100: the model will train on minibatches of 100 examples at each step
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, # dictionary
        y=train_labels,
        batch_size=100,
        shuffle=True
    )
    # step 4.2: use mnist_classifier from tf.estimator.Estimator which assign cnn_model_fn to train the model
    # Trains a model given training data input_fn.
    # steps: Number of steps for which to train model.
    # hooks: List of `SessionRunHook` subclass instances. Used for callbacks inside the training loop.
    # we pass logging_hook to the hooks argument, so that it will be triggered during training
    mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    # step 5: evaluate the model. evaluate the model to determine its accuracy on the MNIST test set
    # step 5.1: define eval_input_fn dictionary
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    # step 5.2: recall evaluation method from Estimator class
    # evaluate method returns a dict containing the evaluation metrics specified in `model_fn` keyed
    # by name, as well as an entry `global_step`
    eval_results = mnist_classifier.evaluate(eval_input_fn=eval_input_fn)
    # step 5.3: print evaluation result
    print eval_results



if __name__ == "__main__":
    # tf.app is just a generic entry point script, which runs the program with an optional 'main' function and 'argv' list
    # it is nothing to do with neural networks and it just calls the main function, passing through any arguments to it
    tf.app.run()