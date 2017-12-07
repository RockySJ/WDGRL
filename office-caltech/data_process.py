import os
import numpy as np
from scipy.io import loadmat
import cPickle as pkl


featuresToUse = "CaffeNet4096"  # surf, CaffeNet4096, GoogleNet1024

# n_split 800 for amazon
# n_split 1000 for caltech10
# n_split 100 for dslr
# n_split 200 for webcam
n_split = 200
domain_name = 'webcam'
domain_data = loadmat(os.path.join("features", featuresToUse, domain_name + '.mat'))
domain_feature = domain_data['fts'].astype(float)
domain_labels = domain_data['labels'].ravel() - 1


def train_valid_split(x, y, split_num):
    data = [x, y]
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    data = [d[shuffle_index] for d in data]
    train_data = data[0][:split_num]
    test_data = data[0][split_num:]
    train_labels = data[1][:split_num]
    test_labels = data[1][split_num:]
    return train_data, test_data, train_labels, test_labels


train_data, test_data, train_labels, test_labels = train_valid_split(domain_feature, domain_labels, n_split)

print train_data.shape
print test_data.shape

with open('pkl/' + domain_name + '_' + featuresToUse[-4:] + '.pkl', 'w') as f:
    domain = {'train': train_data, 'test': test_data, 'train_labels': train_labels, 'test_labels': test_labels}
    pkl.dump(domain, f)

