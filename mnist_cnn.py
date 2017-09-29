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
    # step 6.3: set dropout strategy for tf.layers when train the model


    # step 7: logits layer


if __name__ == "__main__":
    # tf.app is just a generic entry point script, which runs the program with an optional 'main' function and 'argv' list
    # it is nothing to do with neural networks and it just calls the main function, passing through any arguments to it
    tf.app.run()