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
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)), reduction_indices=[1]) # reduction_indices is the old (deprecated) name for axis.
# step 4: assign optimizer for training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# step 5: active session in an InteractiveSession
sess = tf.InteractiveSession()
# step 6: initialize all variables and launch the model by run()
tf.global_variable_initializer().run()
# step 7: train for 1000 steps
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)