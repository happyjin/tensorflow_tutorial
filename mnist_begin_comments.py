import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# The usage for Tensorflow programs is to first create a grap and then launch it in a session
######## Build the computation graph: is a series of TensorFlow operations arranged into a graph of nodes
# step 1: active session in an InteractiveSession
# InteractiveSession class, which makes TensorFlow more flexible about how you structure your code
sess = tf.InteractiveSession()

# step 2: creating nodes for the input images and target output classes
# x and y_ (placeholder) are not specific values until ask tensorflow to run a computation
x = tf.placeholder(tf.float32, shape=[None, 784]) # 2d tensor of floating point numbers
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # None can be of any size, corresponding to the batch size

# step 3: define the weights W and b in Variables
# A Variable is a value that lives in TensorFlow's computation graph. It can be used and even modified by the computation.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# step 4: initialize Variables using session
# Before Variables can be used within a session, they must be initialized using that session.
# this step takes the initial values that have already been specified and assigns them to each Variable. This can be
# done for all Variables at onece.
sess.run(tf.global_variables_initializer())

# step 5: implement the regression model
y = tf.matmul(x, W) + b

# step 6: implement the loss function using cross_entropy
# invoke cross_entropy from tf is more numerically stable
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y))

# step 7: assign training strategy in this case, Steep Gradient Descent
# use SGD with a step length of 0.5 to descend the cross entropy.
# essentially, Tensorflow did is to add new operations to the computation graph. These operations included ones to
# compute gradients, compute parameter update steps, and apply update steps to the parameters
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# step 8: train the model can be accomplished by repeatedly running train_step
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    # when run the returned operation train_step, we will apply the gradient descent updates to the parameters
    # using feed_dict to replace the placeholder tensors x and y_ with the training examples
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# evaluate the model
# step 1: count correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# step 2: compute accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# step 3: print accuracy from graph, we need to apply sess.run in order to print result of test dataset
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))