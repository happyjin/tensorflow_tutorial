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

# step 7