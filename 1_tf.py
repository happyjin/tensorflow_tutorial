import tensorflow as tf

# define constant in tf
# Constants are initialized when you call tf.constant, and their value can never change.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# To actually evaluate the nodes, we must run the computational graph within a session
sess = tf.Session()
print sess.run([node1, node2])

node3 = tf.add(node1, node2)
print("node3:", node3)
print sess.run(node3)

# A placeholder is a promise to provide a value later.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)
print adder_node

# run method to feed concrete values to the placeholders
print sess.run(adder_node, {a:3, b:4.5})
print sess.run(adder_node, {a:[1, 3], b:[2, 4]})

add_and_triple = adder_node * 3
print sess.run(add_and_triple, {a: 3, b: 4.5})

# To make the model trainable, we need to be able to modify the graph to get new outputs
# with the same input. Variables allow us to add trainable parameters to a graph.
# A variable is initialized to the value provided to tf.Variable
# but can be changed using operations like tf.assign
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# variables are not initialized when you call tf.Variable
# To initialize all the variables in a TensorFlow program,
# you must explicitly call a special operation as follows
# init is a handle to the TensorFlow sub-graph that
# initializes all the global variables. Until we call sess.run,
# the variables are uninitialized.
init = tf.global_variables_initializer()
sess.run(init)

print sess.run(linear_model, {x: [1, 2, 3, 4]})

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]})

# improve this manually by reassigning the values of W and b to the perfect values
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]})

###
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})
print sess.run([W, b])