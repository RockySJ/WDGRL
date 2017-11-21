import tensorflow as tf
import utils
import numpy as np
from sklearn.datasets import make_blobs
from flip_gradient import flip_gradient

xs, ys = make_blobs(1000, centers=[[0, 0], [0, 10]], cluster_std=1.5)
xt, yt = make_blobs(1000, centers=[[50, -20], [50, -10]], cluster_std=1.5)

dann_weight = 1
grl_lambda = 1

l2_param = 1e-5
lr = 1e-3
num_step = 5000
batch_size = 64
tf.set_random_seed(0)

n_input = xs.shape[1]
num_class = 2
n_hidden = [20]

with tf.name_scope('input'):
    X = tf.placeholder(dtype=tf.float32)
    y_true = tf.placeholder(dtype=tf.int32)
    train_flag = tf.placeholder(dtype=tf.bool)
    y_true_one_hot = tf.one_hot(y_true, num_class)

h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='dense')

with tf.name_scope('slice_data'):
    h1_s = tf.cond(train_flag, lambda: tf.slice(h1, [0, 0], [batch_size / 2, -1]), lambda: h1)
    h1_t = tf.cond(train_flag, lambda: tf.slice(h1, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [batch_size / 2, -1]), lambda: y_true_one_hot)

with tf.name_scope('dann'):
    d_label = tf.concat(values=[tf.zeros(batch_size / 2, dtype=tf.int32), tf.ones(batch_size / 2, dtype=tf.int32)], axis=0)
    d_label_one_hot = tf.one_hot(d_label, 2)
    h1_grl = flip_gradient(h1, grl_lambda)
    W_domain = tf.Variable(tf.truncated_normal([n_hidden[-1], 2], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='dann_weight')
    b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
    d_logit = tf.matmul(h1_grl, W_domain) + b_domain
    d_softmax = tf.nn.softmax(d_logit)
    domain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(d_label_one_hot, 1), tf.argmax(d_softmax, 1)), tf.float32))
    domain_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=d_label_one_hot))

with tf.name_scope('classifier'):
    W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')
    b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
    pred_logit = tf.matmul(h1_s, W_clf) + b_clf
    pred_softmax = tf.nn.softmax(pred_logit)
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_softmax, 1)), tf.float32))

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + dann_weight * domain_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size / 2, shuffle=False)
    T_batches = utils.batch_generator([xt, yt], batch_size / 2, shuffle=False)

    for i in range(num_step):
        xs_batch, ys_batch = S_batches.next()
        xt_batch, yt_batch = T_batches.next()
        xb = np.vstack([xs_batch, xt_batch])
        yb = np.hstack([ys_batch, yt_batch])
        _, d_acc, d_loss = sess.run([train_op, domain_acc, domain_loss], feed_dict={X: xb, y_true: yb, train_flag: True})
        if i % 1 == 0:
            acc_xs, c_loss_xs = sess.run([clf_acc, clf_loss], feed_dict={X: xs, y_true: ys, train_flag: False})
            acc_xt, c_loss_xt = sess.run([clf_acc, clf_loss], feed_dict={X: xt, y_true: yt, train_flag: False})
            print 'step: ', i
            print 'domain accuracy: %f, domain loss: %f' % (d_acc, d_loss)
            print 'Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt)
            print 'Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt)
