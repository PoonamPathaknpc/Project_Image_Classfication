import tensorflow as tf
import matplotlib.pyplog as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print("Size of:")
print("- Training_set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))


sess = tf.InteractiveSession()


########## Hyper Parameter ###########
learning_rate = 1e-4
batch_Size = 100
featureMap_1 = 20
featureMap_2 = 40
dropRate = 0.5
fc1_ = 300
fc2_ = 300
Iteration = 10000
#####################




##########################################

########## CNN functions ##########
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

##########################################



x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

############ Convolution Layer 1 ############
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5,5,1,featureMap_1])
b_conv1 = bias_variable([featureMap_1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#############################################

############ Convolution Layer 2 ############
W_conv2 = weight_variable([5,5,featureMap_1,featureMap_2])
b_conv2 = bias_variable([featureMap_2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


############ Fully Connected Layer 1 ############
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*featureMap_2])

W_fc1 = weight_variable([7*7*featureMap_2, fc1_])
b_fc1 = bias_variable([fc1_])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#############################################


############ Fully Connected Layer 2 ############

W_fc2 = weight_variable([fc1_, fc2_])
b_fc2 = bias_variable([fc2_])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#############################################

############ SoftMax Layer ############

W_softMax = weight_variable([fc2_, 10])
b_softMax = bias_variable([10])

y_conv = tf.matmul(h_fc2_drop, W_softMax) + b_softMax
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
