from glob import glob
import random
from gatherData import gatherAllXYNoWindow, gatherAllXY, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN

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

window_size = 5
def _gather(fn):
    return gatherAllXY(fn, 5)

pool = mp.Pool()
train_data = pool.map(_gather, train_data_files)
# Xtrain = np.vstack([x for (xs, ys) in train_data for x in xs])
# Ytrain = np.vstack([y for (xs, ys) in train_data for y in ys])
# del train_data

validation_data = pool.map(_gather, validation_data_files)
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

print '... Building model'
# model = ChainCRF()
# Xs, Ys = zip(*train_data)

# plt.plt(ssvm.loss_curve_)
# plt.show()

import pycrfsuite

# Changes

trainer = pycrfsuite.Trainer(verbose=True, algorithm='lbfgs')
for xseq, yseq in train_data:
    trainer.append([dict([(str(k), v) for k, v in enumerate(x)]) for x in xseq],
                    [str(int(y)) for y in yseq])

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 75,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('crfsuite_basic_crf.model')
