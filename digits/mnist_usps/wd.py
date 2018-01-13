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
num_steps = 20000

wd_param = 0.01
D_train_num = 5
gp_param = 10
lr_wd_D = 1e-4

with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    train_flag = tf.placeholder(tf.bool)

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

with tf.name_scope('slice_data'):
    h_s = tf.cond(train_flag, lambda: tf.slice(h_fc1, [0, 0], [batch_size / 2, -1]), lambda: h_fc1)
    h_t = tf.cond(train_flag, lambda: tf.slice(h_fc1, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h_fc1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_, [0, 0], [batch_size / 2, -1]), lambda: y_)

with tf.name_scope('classifier'):
    W_fc2 = utils.weight_variable([1024, 10], 'fc2_weight')
    b_fc2 = utils.bias_variable([10], 'fc2_bias')
    pred_logit = tf.matmul(h_fc1, W_fc2) + b_fc2
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=y_))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(pred_logit, 1)), tf.float32))

alpha = tf.random_uniform(shape=[batch_size / 2, 1], minval=0., maxval=1.)
differences = h_s - h_t
interpolates = h_t + (alpha*differences)
h2_whole = tf.concat([h_fc1, interpolates], 0)

with tf.name_scope('critic'):
    critic_h1 = utils.fc_layer(h2_whole, 1024, 100, layer_name='critic_h1')
    critic_out = utils.fc_layer(critic_h1, 100, 1, layer_name='critic_h2', act=tf.identity)

critic_s = tf.cond(train_flag, lambda: tf.slice(critic_out, [0, 0], [batch_size / 2, -1]), lambda: critic_out)
critic_t = tf.cond(train_flag, lambda: tf.slice(critic_out, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: critic_out)
wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))
gradients = tf.gradients(critic_out, [h2_whole])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
theta_C = [v for v in tf.global_variables() if 'classifier' in v.name]
theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
theta_G = [v for v in tf.global_variables() if 'generator' in v.name]
wd_d_op = tf.train.AdamOptimizer(lr_wd_D).minimize(-wd_loss+gp_param*gradient_penalty, var_list=theta_D)

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + wd_param * wd_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=theta_G + theta_C)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size / 2)
    T_batches = utils.batch_generator([xt, yt], batch_size / 2)

    for i in range(num_steps):
        xs_batch, ys_batch = S_batches.next()
        xt_batch, yt_batch = T_batches.next()
        xb = np.vstack([xs_batch, xt_batch])
        yb = np.vstack([ys_batch, yt_batch])
        for _ in range(D_train_num):
            sess.run(wd_d_op, feed_dict={x: xb, train_flag: True})

        sess.run(train_op, feed_dict={x: xb, y_: yb, train_flag: True})

        if i % 100 == 0:
            acc, clf_ls = sess.run([clf_acc, clf_loss], feed_dict={x: xs_test, y_: ys_test})
            acc_m, clf_ls_m = sess.run([clf_acc, clf_loss], feed_dict={x: xt_test, y_: yt_test})
            print 'step', i
            print 'source classifier loss: %f, source accuracy: %f' % (clf_ls, acc)
            print 'target classifier loss: %f, target accuracy: %f ' % (clf_ls_m, acc_m)
