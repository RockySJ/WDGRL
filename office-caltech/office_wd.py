import tensorflow as tf
import utils
import numpy as np
from functools import partial
from flip_gradient import flip_gradient

data_dir = './pkl/'
feature_type = '4096'
source_name = 'amazon'
target_name = 'caltech10'
xs, ys, xt, yt, xt_test, yt_test = utils.load_office(source_name, target_name, data_dir, feature_type)

wd_param = 0.1
gp_param = 10
lr_wd_D = 1e-3
D_train_num = 10

l2_param = 1e-5
lr = 1e-4
batch_size = 64
num_steps = 1200
num_class = 10
n_input = 4096
n_hidden = [500, 100]

with tf.name_scope('input'):
    X = tf.placeholder(dtype=tf.float32)
    y_true = tf.placeholder(dtype=tf.int32)
    train_flag = tf.placeholder(dtype=tf.bool)
    y_true_one_hot = tf.one_hot(y_true, num_class)

with tf.name_scope('generator'):
    h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1')
    h2 = utils.fc_layer(h1, n_hidden[0], n_hidden[1], layer_name='hidden2')

with tf.name_scope('slice_data'):
    h2_s = tf.cond(train_flag, lambda: tf.slice(h2, [0, 0], [batch_size / 2, -1]), lambda: h2)
    h2_t = tf.cond(train_flag, lambda: tf.slice(h2, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h2)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [batch_size / 2, -1]), lambda: y_true_one_hot)

with tf.name_scope('classifier'):
    W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')
    b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
    pred_logit = tf.matmul(h2_s, W_clf) + b_clf
    pred_softmax = tf.nn.softmax(pred_logit)
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true_one_hot, 1), tf.argmax(pred_softmax, 1)), tf.float32))

alpha = tf.random_uniform(shape=[batch_size / 2, 1], minval=0., maxval=1.)
differences = h2_s - h2_t
interpolates = h2_t + (alpha*differences)
h2_whole = tf.concat([h2, interpolates], 0)

with tf.name_scope('critic'):
    critic_h1 = utils.fc_layer(h2_whole, n_hidden[-1], 100, layer_name='critic_h1')
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
        yb = np.hstack([ys_batch, yt_batch])

        for _ in range(D_train_num):
            sess.run(wd_d_op, feed_dict={X: xb, train_flag: True})

        sess.run(train_op, feed_dict={X: xb, y_true: yb, train_flag: True})

        if i % 10 == 0:
            acc, clf_ls = sess.run([clf_acc, clf_loss], feed_dict={X: xs, y_true: ys, train_flag: False})
            acc_m, clf_ls_m = sess.run([clf_acc, clf_loss], feed_dict={X: xt_test, y_true: yt_test, train_flag: False})
            print 'step', i
            print 'source classifier loss: %f, source accuracy: %f' % (clf_ls, acc)
            print 'target classifier loss: %f, target accuracy: %f ' % (clf_ls_m, acc_m)
