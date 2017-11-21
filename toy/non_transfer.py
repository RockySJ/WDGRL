import tensorflow as tf
import utils
from sklearn.datasets import make_blobs

xs, ys = make_blobs(1000, centers=[[0, 0], [0, 10]], cluster_std=1.5)
xt, yt = make_blobs(1000, centers=[[50, -20], [50, -10]], cluster_std=1.5)

utils.plot_data(xs, ys, xt, yt)

l2_param = 1e-5
lr = 1e-3
training_epoch = 1000
tf.set_random_seed(0)

n_input = xs.shape[1]
num_class = 2
n_hidden = [20]

X = tf.placeholder(dtype=tf.float32)
y_true = tf.placeholder(dtype=tf.int32)
y_true_one_hot = tf.one_hot(y_true, num_class)

h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='dense')

W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')
b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')
pred_logit = tf.matmul(h1, W_clf) + b_clf
pred_softmax = tf.nn.softmax(pred_logit)
clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=y_true_one_hot))
clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true_one_hot, 1), tf.argmax(pred_softmax, 1)), tf.float32))

all_variables = tf.trainable_variables()
l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
total_loss = clf_loss + l2_loss
train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    S_batches = utils.batch_generator([xs, ys], 32)
    total_batch = int(xs.shape[0] / 32)

    for epoch in range(training_epoch):
        for i in range(total_batch):
            xs_batch, ys_batch = S_batches.next()
            sess.run([train_op], feed_dict={X: xs_batch, y_true: ys_batch})
        acc_xs, c_loss_xs = sess.run([clf_acc, clf_loss], feed_dict={X: xs, y_true: ys})
        acc_xt, c_loss_xt = sess.run([clf_acc, clf_loss], feed_dict={X: xt, y_true: yt})
        print 'epoch: ', epoch
        print 'Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt)
        print 'Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt)
