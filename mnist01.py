import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()


########## Hyper Parameter ###########
learning_rate = 1e-4
batch_Size = 100
featureMap = 32
dropRate = 1.0
fc1_ = 200
Iteration = 1000
#####################

def variable_summaries(var):
  """Visual Representation consisting 3 layers of CNN (Convolution Neural Network) """
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)



def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

############ Convolution Layer 1 ############
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5,5,1,featureMap])
b_conv1 = bias_variable([featureMap])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#############################################


############ Fully Connected Layer 1 ############
h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*featureMap])

W_fc1 = weight_variable([14*14*featureMap, fc1_])
b_fc1 = bias_variable([fc1_])

h_fc1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#############################################


############ Fully Connected Layer 2 ############
W_fc2 = weight_variable([fc1_, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
#############################################

cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(Iteration):
	batch = mnist.train.next_batch(batch_Size)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob:dropRate})

print("test accuracy %g "%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
