import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

### weight initializaiton
# function in order to initial weight variable
def weight_variable(shape):
    # initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# function in order to initial bias variable
def bias_variable(shape):
    # initialize them with a slightly positive initial bias to avoid "dead neurons" in practical
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


### Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    # step 1: define placeholder in order to assign data
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    ### build the CNN model
    # step 2: the first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # reshape the data into 4D tensor
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # [-1, image_width, image_height, number of color channels]

    # step 2.2: convolve x_image with the weight tensor, add the bias, apply ReLU funtion, and finally max pool
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # step 3: the second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    # step 3.2: convolve x_image with the weight tensor, add the bias, apply ReLU funtion, and finally max pool
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # step 4: Densely Connected Layer (fully-connected layer)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # step 5: Dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # step 6: Readout Layer, just like for the one layer softmax regression above.
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    ### train and evaluate the model
    # step 1: set how to account loss and accuracy using cross_entropy in this case
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ### save graph includes Variables using saver object
    all_saver = tf.train.Saver()

    # step 2: pass real data(for placeholder) into model and train the model using session
    with tf.Session() as sess:
        # step 2.1: initialize all variables before using them
        sess.run(tf.global_variables_initializer())
        # step 3: set training maximum training steps
        for i in range(200):
            # step 3.1: set batch_size
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                #print sess.run(tf.trainable_variables()) # print all training variables for this model
                # step 3.3: evaluate the model after every 100 steps
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                # step 3.4: print current result under current batch data
                print('step %d, training accuracy %g' % (i, train_accuracy))
            # step 3.2: train the model and feed the current batch data
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print train_step
        #all_saver.save(sess, 'data.chkp')

        # step 4: print training result that is accuracy
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        # test
        test_sample = mnist.test.images[0].reshape(1, -1)
        test_label = mnist.test.labels[0].reshape(1, -1)
        #print sess.run(correct_prediction, feed_dict={x: test_sample, y_: test_label})