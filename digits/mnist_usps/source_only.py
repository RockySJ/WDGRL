import utils
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('data', one_hot=True)
data_dir = 'data/usps_28x28.pkl'
xt, yt, xt_test, yt_test = utils.load_usps(data_dir, one_hot=True, flatten=True)
xs = mnist.train.images
ys = mnist.train.labels
xs_test = mnist.test.images
ys_test = mnist.test.labels

l2_param = 1e-5
lr = 1e-4
batch_size = 64
num_steps = 10000

with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('feature_generator'):
    W_conv1 = utils.weight_variable([5, 5, 1, 32], 'conv1_weight')
    b_conv1 = utils.bias_variable([32], 'conv1_bias')
    h_conv1 = tf.nn.relu(utils.conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = utils.max_pool_2x2(h_conv1)

    W_conv2 = utils.weight_variable([5, 5, 32, 64], 'conv2_weight')
    b_conv2 = utils.weight_variable([64], 'conv2_bias')
    h_conv2 = tf.nn.relu(utils.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = utils.max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    W_fc1 = utils.weight_variable([7*7*64, 1024], 'fc1_weight')
    b_fc1 = utils.bias_variable([1024], 'fc1_bias')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

with tf.name_scope('classifier'):
    W_fc2 = utils.weight_variable([1024, 10], 'fc2_weight')
    b_fc2 = utils.bias_variable([10], 'fc2_bias')
    pred_logit = tf.matmul(h_fc1, W_fc2) + b_fc2
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=y_))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(pred_logit, 1)), tf.float32))

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size)

    for i in range(num_steps):
        xs_batch, ys_batch = S_batches.next()
        sess.run(train_op, feed_dict={x: xs_batch, y_: ys_batch})

        if i % 100 == 0:
            acc, clf_ls = sess.run([clf_acc, clf_loss], feed_dict={x: xs_test, y_: ys_test})
            acc_m, clf_ls_m = sess.run([clf_acc, clf_loss], feed_dict={x: xt_test, y_: yt_test})
            print 'step', i
            print 'source classifier loss: %f, source accuracy: %f' % (clf_ls, acc)
            print 'target classifier loss: %f, target accuracy: %f ' % (clf_ls_m, acc_m)
