import custom_crf_base as crf
from glob import glob
import random
from gatherData import allLabels, gatherAllXYNoWindow, gatherAllXY, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN
import numpy as np

import multiprocessing as mp
import os

from sklearn.metrics import confusion_matrix

# import numpy as np
# import matplotlib.pyplot as plt


window_size = 15
maxiter = 500
lamb = 1.0
output = True

# train_num_files = 10
# train_data_files = random.sample(glob(TRAIN_FILE_PATTERN), train_num_files)
train_data_files = glob(TRAIN_FILE_PATTERN)

# validation_num_files = 10
# validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)

print '... Gathering data'

# Diff version
def _gather(fn):
    return gatherAllXY(fn, window_size, diff=True)

pool = mp.Pool()
train_data = pool.map(_gather, train_data_files)
# valid_data = pool.map(_gather, validation_data_files)

train_xs, train_ys = zip(*[(x.T, y) for x, y in train_data])
# valid_xs, valid_ys = zip(*[(x.T, y) for x, y in valid_data])

del train_data
# del valid_data

# import resource
# print resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'Mb'


K, _ = train_xs[0].shape
D = len(allLabels)

print '... Training model'
opt_results, ll_per_iter = crf.learnParameters(train_xs, train_ys, K, D, lamb=lamb, maxiter=maxiter)
# crf.learnSGD(train_xs, train_ys, K, D, lamb=lamb, maxiter=maxiter)

if output:
    theta, gamma = crf.vectorToParams(opt_results[0], K, D)

    results_dir = 'results/crf_train_%d_windowsize_%d_lamb_%f_maxiter_%d/' % (len(train_xs), window_size, lamb, maxiter)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.save(results_dir + 'theta.npy', theta)
    np.save(results_dir + 'gamma.npy', gamma)
    np.save(results_dir + 'll_per_iter.npy', ll_per_iter)

# cm = np.zeros((D, D))
# for i in range(validation_num_files):
#     pred = crf.posteriorMAP(valid_xs[i], theta, gamma, D)
#     cm += confusion_matrix(valid_ys[i], pred, labels=range(D))

# np.save('conf_mat_crf_10_hipped.npy', cm)
