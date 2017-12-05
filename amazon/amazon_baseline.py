import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack
from flip_gradient import flip_gradient
from functools import partial


data_folder = './data/'
source_name = 'dvd'
target_name = 'electronics'
xs, ys, xt, yt, xt_test, yt_test = utils.load_amazon(source_name, target_name, data_folder, verbose=True)

save_model = False
save_path = 'mmd_model/dvd_electronics'
log_dir = './log'

# mmd coefficient
mmd_param = 0.5
# dann coefficient
grl_lambda = 0
dann_param = 0
# coral coefficient
coral_param = 0

batch_size = 64
l2_param = 1e-4
lr = 1e-4
num_step = 10000
num_class = 2
n_input = xs.shape[1]
n_hidden = [500]

tf.set_random_seed(0)
np.random.seed(0)

with tf.name_scope('input'):
    X = tf.sparse_placeholder(dtype=tf.float32)
    y_true = tf.placeholder(dtype=tf.int32)
    train_flag = tf.placeholder(dtype=tf.bool)
    y_true_one_hot = tf.one_hot(y_true, num_class)

h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='sparse')

with tf.name_scope('slice_data'):
    h1_s = tf.cond(train_flag, lambda: tf.slice(h1, [0, 0], [batch_size / 2, -1]), lambda: h1)
    h1_t = tf.cond(train_flag, lambda: tf.slice(h1, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [batch_size / 2, -1]), lambda: y_true_one_hot)


with tf.name_scope('classifier'):
    W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')
    b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
    pred_logit = tf.matmul(h1_s, W_clf) + b_clf
    pred_softmax = tf.nn.softmax(pred_logit)
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_softmax, 1)), tf.float32))
    clf_loss_sum = tf.summary.scalar('clf_loss', clf_loss)
    clf_acc_sum = tf.summary.scalar('clf_acc', clf_acc)

with tf.name_scope('dann'):
    d_label = tf.concat(values=[tf.zeros(batch_size / 2, dtype=tf.int32), tf.ones(batch_size / 2, dtype=tf.int32)], axis=0)
    d_label_one_hot = tf.one_hot(d_label, 2)
    h1_grl = flip_gradient(h1, grl_lambda)
    h_dann_1 = utils.fc_layer(h1_grl, n_hidden[-1], 100, layer_name='dann_fc_1')
    W_domain = tf.Variable(tf.truncated_normal([100, 2], stddev=1. / tf.sqrt(100 / 2.)), name='dann_weight')
    b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
    d_logit = tf.matmul(h_dann_1, W_domain) + b_domain
    d_softmax = tf.nn.softmax(d_logit)
    domain_loss = dann_param * tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=d_label_one_hot))
    domain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(d_label_one_hot, 1), tf.argmax(d_softmax, 1)), tf.float32))
    tf.summary.scalar('domain_loss', domain_loss)
    tf.summary.scalar('domain_acc', domain_acc)

with tf.name_scope('mmd'):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
    loss_value = utils.maximum_mean_discrepancy(h1_s, h1_t, kernel=gaussian_kernel)
    mmd_loss = mmd_param * tf.maximum(1e-4, loss_value)
    assert_op = tf.Assert(tf.is_finite(mmd_loss), [mmd_loss])
    tf.summary.scalar('mmd_loss', mmd_loss)

with tf.name_scope('coral_loss'):
    _D_s = tf.reduce_sum(h1_s, axis=0, keep_dims=True)
    _D_t = tf.reduce_sum(h1_t, axis=0, keep_dims=True)
    C_s = (tf.matmul(tf.transpose(h1_s), h1_s) - tf.matmul(tf.transpose(_D_s), _D_s) / batch_size/2) / (batch_size/2 - 1)
    C_t = (tf.matmul(tf.transpose(h1_t), h1_t) - tf.matmul(tf.transpose(_D_t), _D_t) / batch_size/2) / (batch_size/2 - 1)
    coral_loss = coral_param * tf.nn.l2_loss(C_s - C_t)
    tf.summary.scalar('coral_loss', coral_loss)

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + mmd_loss + domain_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

merged = tf.summary.merge_all()
test_merged = tf.summary.merge([clf_loss_sum, clf_acc_sum])
saver = tf.train.Saver(max_to_keep=100)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], batch_size / 2, shuffle=True)
    T_batches = utils.batch_generator([xt, yt], batch_size / 2, shuffle=True)

    for i in range(num_step):
        xs_batch_csr, ys_batch = S_batches.next()
        xt_batch_csr, yt_batch = T_batches.next()
        batch_csr = vstack([xs_batch_csr, xt_batch_csr])
        xb = utils.csr_2_sparse_tensor_tuple(batch_csr)
        yb = np.hstack([ys_batch, yt_batch])
        _, train_summary = sess.run([train_op, merged], feed_dict={X: xb, y_true: yb, train_flag: True})
        train_writer.add_summary(train_summary, global_step=i)

        if i % 30 == 0:
            whole_xs_stt = utils.csr_2_sparse_tensor_tuple(xs)
            acc_xs, c_loss_xs = sess.run([clf_acc, clf_loss], feed_dict={X: whole_xs_stt, y_true: ys, train_flag: False})
            whole_xt_stt = utils.csr_2_sparse_tensor_tuple(xt_test)
            test_summary, acc_xt, c_loss_xt = sess.run([test_merged, clf_acc, clf_loss], feed_dict={X: whole_xt_stt, y_true: yt_test, train_flag: False})
            test_writer.add_summary(test_summary, global_step=i)
            print 'step: ', i
            print 'Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt)
            print 'Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt)

            if save_model:
                saver.save(sess, save_path, global_step=i)
