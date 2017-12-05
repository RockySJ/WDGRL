import numpy as np
import tensorflow as tf
import utils
from sklearn.manifold import TSNE

data_folder = './data/'
source_name = 'dvd'
target_name = 'electronics'
xs, ys, xt, yt, xt_test, yt_test = utils.load_amazon(source_name, target_name, data_folder, verbose=True)

load_path = 'wgan_model/best/dvd_electronics'
n_input = xs.shape[1]
n_hidden = [500]

X = tf.sparse_placeholder(dtype=tf.float32)

with tf.name_scope('generator'):
    h1 = utils.fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='sparse')

theta_G = [v for v in tf.global_variables() if 'hidden1' in v.name]
saver = tf.train.Saver(var_list=theta_G)
sess = tf.Session()
saver.restore(sess, load_path)
whole_xs_stt = utils.csr_2_sparse_tensor_tuple(xs)
whole_xt_stt = utils.csr_2_sparse_tensor_tuple(xt)
hs = sess.run(h1, feed_dict={X: whole_xs_stt})
ht = sess.run(h1, feed_dict={X: whole_xt_stt})

h_both = np.vstack([hs, ht])
y = np.hstack([ys, yt])
source_num = hs.shape[0]

tsne = TSNE(perplexity=30, n_components=2, n_iter=3300)
source_only_tsne = tsne.fit_transform(h_both)
utils.plot_embedding(source_only_tsne, y, source_num, 'wgan_tsne.pdf')

