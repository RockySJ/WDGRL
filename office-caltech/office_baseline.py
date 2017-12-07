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

# mmd coefficient
mmd_param = 0.3
# dann coefficient
grl_lambda = 0
dann_param = 0
# coral coefficient
coral_param = 0

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

with tf.name_scope('dann'):
    d_label = tf.concat(values=[tf.zeros(batch_size / 2, dtype=tf.int32), tf.ones(batch_size / 2, dtype=tf.int32)], axis=0)
    d_label_one_hot = tf.one_hot(d_label, 2)

    h2_grl = flip_gradient(h2, grl_lambda)
    dann_fc1 = utils.fc_layer(h2_grl, n_hidden[-1], 100, layer_name='dann_fc_1')
    W_domain = tf.Variable(tf.truncated_normal([100, 2], stddev=1. / tf.sqrt(100 / 2.)), name='dann_weight')
    b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
    d_logit = tf.matmul(dann_fc1, W_domain) + b_domain
    d_softmax = tf.nn.softmax(d_logit)
    domain_loss = dann_param * tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=d_label_one_hot))

    domain_loss_sum = tf.summary.scalar('domain_loss', domain_loss)

with tf.name_scope('mmd'):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = utils.maximum_mean_discrepancy(h2_s, h2_t, kernel=gaussian_kernel)
    mmd_loss = mmd_param * tf.maximum(1e-4, loss_value)

with tf.name_scope('coral_loss'):
    _D_s = tf.reduce_sum(h2_s, axis=0, keep_dims=True)
    _D_t = tf.reduce_sum(h2_t, axis=0, keep_dims=True)
    C_s = (tf.matmul(tf.transpose(h2_s), h2_s) - tf.matmul(tf.transpose(_D_s), _D_s) / batch_size/2) / (batch_size/2 - 1)
    C_t = (tf.matmul(tf.transpose(h2_t), h2_t) - tf.matmul(tf.transpose(_D_t), _D_t) / batch_size/2) / (batch_size/2 - 1)
    coral_loss = coral_param * tf.nn.l2_loss(C_s - C_t)

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + mmd_loss + coral_loss + domain_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size / 2)
    T_batches = utils.batch_generator([xt, yt], batch_size / 2)

    for i in range(num_steps):
        xs_batch, ys_batch = S_batches.next()
        xt_batch, yt_batch = T_batches.next()
        xb = np.vstack([xs_batch, xt_batch])
        yb = np.hstack([ys_batch, yt_batch])
        sess.run(train_op, feed_dict={X: xb, y_true: yb, train_flag: True})

        if i % 10 == 0:
            acc, clf_ls = sess.run([clf_acc, clf_loss], feed_dict={X: xs, y_true: ys, train_flag: False})
            acc_m, clf_ls_m = sess.run([clf_acc, clf_loss], feed_dict={X: xt_test, y_true: yt_test, train_flag: False})
            print 'step', i
            print 'source classifier loss: %f, source accuracy: %f' % (clf_ls, acc)
            print 'target classifier loss: %f, target accuracy: %f ' % (clf_ls_m, acc_m)
