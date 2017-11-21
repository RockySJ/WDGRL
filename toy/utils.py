import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense'):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)
        return activations


def plot_data(xs, ys, xt, yt):
    ys_pos_index = np.where(ys == 1)[0]
    ys_neg_index = np.where(ys == 0)[0]
    xs_pos = xs[ys_pos_index]
    xs_neg = xs[ys_neg_index]
    yt_pos_index = np.where(yt == 1)[0]
    yt_neg_index = np.where(yt == 0)[0]
    xt_pos = xt[yt_pos_index]
    xt_neg = xt[yt_neg_index]
    plt.scatter(xs_pos[:, 0], xs_pos[:, 1], c='r', s=4, alpha=0.7, label='source positive')
    plt.scatter(xs_neg[:, 0], xs_neg[:, 1], c='b', s=4, alpha=0.7, label='source negative')
    plt.scatter(xt_pos[:, 0], xt_pos[:, 1], c='g', s=4, alpha=0.7, label='target positive')
    plt.scatter(xt_neg[:, 0], xt_neg[:, 1], c='black', s=4, alpha=0.7, label='target negative')
    plt.legend()
    plt.show()
