from glob import glob
import random
from gatherData import gatherAllXYNoWindow, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN

import numpy as np
import matplotlib.pyplot as plt

from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

import multiprocessing as mp


train_num_files = 50
train_file_pattern = '/users/skainswo/data/skainswo/chalearn/train/Sample*_data.mat'
train_data_files = random.sample(glob(train_file_pattern), train_num_files)

validation_num_files = 5
validation_file_pattern = '/users/skainswo/data/skainswo/chalearn/validation/Sample*_data.mat'
validation_data_files = random.sample(glob(validation_file_pattern), validation_num_files)


print '... Gathering data'

pool = mp.Pool()
train_data = pool.map(gatherAllXYNoWindow, train_data_files)
# Xtrain = np.vstack([x for (xs, ys) in train_data for x in xs])
# Ytrain = np.vstack([y for (xs, ys) in train_data for y in ys])
# del train_data

validation_data = pool.map(gatherAllXYNoWindow, validation_data_files)
# Xvalid = np.vstack([x for (xs, ys) in validation_data for x in xs])
# Yvalid = np.vstack([y for (xs, ys) in validation_data for y in ys])
# del validation_data

print len(train_data)

# print '... Building model'
# model = ChainCRF()
# ssvm = OneSlackSSVM(model, C=0.1, show_loss_every=100, n_jobs=-1)
# Xs, Ys = zip(*train_data)
# ssvm.fit(Xs, [y.astype(int) for y in Ys])

# plt.plt(ssvm.loss_curve_)
# plt.show()
