import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack


data_folder = './data/'
source_name = 'dvd'
target_name = 'electronics'
xs, ys, xt, yt, xt_test, yt_test = utils.load_amazon(source_name, target_name, data_folder, verbose=True)

save_model = False
save_path = 'wdgrl_model/dvd_electronics'
log_dir = './log'

# hyper-parameter for wdgrl
wd_param = 1
gp_param = 10
lr_wd_D = 1e-4
D_train_num = 5

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

with tf.name_scope('generator'):
    h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='sparse')

with tf.name_scope('slice_data'):
    h1_s = tf.cond(train_flag, lambda: tf.slice(h1, [0, 0], [batch_size / 2, -1]), lambda: h1)
    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [batch_size / 2, -1]), lambda: y_true_one_hot)
    h1_t = tf.cond(train_flag, lambda: tf.slice(h1, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: h1)

with tf.name_scope('classifier'):
    W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')
    b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
    pred_logit = tf.matmul(h1_s, W_clf) + b_clf
    pred_softmax = tf.nn.softmax(pred_logit)
    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))
    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_softmax, 1)), tf.float32))
    clf_loss_sum = tf.summary.scalar('clf_loss', clf_loss)
    clf_acc_sum = tf.summary.scalar('clf_acc', clf_acc)

alpha = tf.random_uniform(shape=[batch_size / 2, 1], minval=0., maxval=1.)
differences = h1_s - h1_t
interpolates = h1_t + (alpha*differences)
h1_whole = tf.concat([h1, interpolates], 0)

with tf.name_scope('critic'):
    critic_h1 = utils.fc_layer(h1_whole, n_hidden[-1], 100, layer_name='critic_h1')
    critic_out = utils.fc_layer(critic_h1, 100, 1, layer_name='critic_h2', act=tf.identity)

critic_s = tf.cond(train_flag, lambda: tf.slice(critic_out, [0, 0], [batch_size / 2, -1]), lambda: critic_out)
critic_t = tf.cond(train_flag, lambda: tf.slice(critic_out, [batch_size / 2, 0], [batch_size / 2, -1]), lambda: critic_out)
wd_loss = (tf.reduce_mean(critic_s) - tf.reduce_mean(critic_t))
tf.summary.scalar('wd_loss', wd_loss)
gradients = tf.gradients(critic_out, [h1_whole])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
tf.summary.scalar('gradient_penalty', gradient_penalty)
theta_C = [v for v in tf.global_variables() if 'classifier' in v.name]
theta_D = [v for v in tf.global_variables() if 'critic' in v.name]
theta_G = [v for v in tf.global_variables() if 'generator' in v.name]
wd_d_op = tf.train.AdamOptimizer(lr_wd_D).minimize(-wd_loss+gp_param*gradient_penalty, var_list=theta_D)
all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss + wd_param * wd_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list=theta_G + theta_C)

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
        for _ in range(D_train_num):
            sess.run([wd_d_op], feed_dict={X: xb, train_flag: True})
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
