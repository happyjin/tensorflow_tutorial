import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print mnist.train
print mnist.test
print mnist.validation
print mnist.train.images.shape
print mnist.train.labels.shape
print type(mnist.train.labels)

# step 1: define parameters
x = tf.placeholder(tf.float32, [None, 784]) # None means that a dimension can be of any length
W = tf.Variable(tf.zeros([784, 10])) # initialize this variable with zeros
b = tf.Variable(tf.zeros([10]))
# step 2: implement model (model with no specific data in it)
y = tf.nn.softmax(tf.matmul(x, W) + b)
# step 3: implement cross-entropy (model with no specific data in it)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y_) # more stable numerically
# step 4: assign optimizer for training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# step 5: active session in an InteractiveSession
sess = tf.InteractiveSession()
# step 6: initialize all variables and launch the model by run()
tf.global_variables_initializer().run()
# step 7: train for 1000 steps
for _ in range(1000):
    # Each step of the loop, we get a "batch" of one hundred random data points from our training set
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # run train_step feeding in the batches data to replace the placeholders
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# evaluate the model
# step 1: count correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# step 2: compute accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# step 3: print accuracy from graph, we need to apply sess.run in order to print result of test dataset
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))