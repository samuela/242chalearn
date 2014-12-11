import custom_crf_base as crf
from glob import glob
import random
from gatherData import allLabels, gatherAllXYNoWindow, gatherAllXY, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN
import numpy as np

import multiprocessing as mp

from sklearn.metrics import confusion_matrix

# import numpy as np
# import matplotlib.pyplot as plt


train_num_files = 10
train_data_files = random.sample(glob(TRAIN_FILE_PATTERN), train_num_files)

validation_num_files = 10
validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)

print '... Gathering data'

window_size = 15
def _gather(fn):
    return gatherAllXY(fn, window_size)

pool = mp.Pool()
train_data = pool.map(_gather, train_data_files)
valid_data = pool.map(_gather, validation_data_files)

train_xs, train_ys = zip(*[(x.T, y) for x, y in train_data])
valid_xs, valid_ys = zip(*[(x.T, y) for x, y in valid_data])

K, _ = train_xs[0].shape
D = len(allLabels)

print '... Training model'
params, nll, d = crf.learnParameters(train_xs, train_ys, K, D, maxiter=150)
# crf.learnSGD(train_xs, train_ys, K, D, maxiter=250)

theta, gamma = crf.vectorToParams(params, K, D)

cm = np.zeros((D, D))
for i in range(validation_num_files):
    pred = crf.posteriorMAP(valid_xs[i], theta, gamma, D)
    cm += confusion_matrix(valid_ys[i], pred, labels=range(D))

np.save('conf_mat_crf_10_hipped.npy', cm)
