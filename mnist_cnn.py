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
    :param features: input data
    :param labels: labels for data
    :param mode:
    :return:
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


if __name__ == "__main__":
    # tf.app is just a generic entry point script, which runs the program with an optional 'main' function and 'argv' list
    # it is nothing to do with neural networks and it just calls the main function, passing through any arguments to it
    tf.app.run()