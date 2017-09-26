import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of square
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01) # choose optimization method
train = optimizer.minimize(loss) # the aim of train is to minimize the loss using optimizer

# training data
x_train = [1, 2, 3, 4]
y_trian = [0, -1, -3, -3]
# training loop
# 1 step: initial all global variables
init = tf.global_variables_initializer()
# 2 step: To actually evaluate the nodes, we must run the computational graph within a session
sess = tf.Session()
# 3 step: run session to evaluate all nodes in a graph
# 3.1 step: run the initial session for initializing all nodes
sess.run(init) # reset values to wrong. There is no placeholder for feeding value
# 3.2 step: run the train session to train all loaded nodes. x and y are placeholder to wait for feeding value
for i in range(1000):
    sess.run(train, {x: x_train, y: y_trian})

# evaluate training accuracy
# when I start sess.run. I can start evaluate the graph with feeding value.
# Since loss contains placeholder x and y then we need to assign x and y
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_trian}) # if you want to display its value of graph then use sess.run
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))