import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

### logging device placement
# step 1: create a graph
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)


# step 2: create a session with log_device_placement set to True in order to find out
# which devices your operations and tensors are assigned to
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# step 3: run the op. and print the result
print sess.run(c)
print '---'

### manuel device placement
# step 1: create a graph
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# step 2: creates a session with log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# step 3: run the op. and print the result
print sess.run(c)
print '---'

### allow software to choose if there not exists device
# step 1: create a graph
with tf.device('/gpu:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# step 2: create a session with allow_soft_placement and log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

# step 3: run the op. and print the result
print sess.run(c)
print '---'

### using multiple GPUs
# step 1: create a graph
c = []
for d in ['/gpu:2', '/gpu:3']:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
    sum = tf.add_n(c)

# step 2: create a session with allow_soft_placement and log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

# step 3: run the op. and print the result
sess.run(sum)