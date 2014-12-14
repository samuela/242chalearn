import custom_crf_base as crf

import random
from glob import glob
from gatherData import gatherAllXY, VALID_FILE_PATTERN
from itertools import groupby
import numpy as np

import multiprocessing as mp

# validation_num_files = 1
# validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)
validation_data_files = glob(VALID_FILE_PATTERN)

print '... Gathering data'

theta = np.load('results/crf_train_393_windowsize_15_lamb_1.000000_maxiter_500_randominit/theta.npy')
gamma = np.load('results/crf_train_393_windowsize_15_lamb_1.000000_maxiter_500_randominit/gamma.npy')

window_size = 15
def _gather(fn):
    return gatherAllXY(fn, window_size)

pool = mp.Pool()
valid_data = pool.map(_gather, validation_data_files)
# valid_xs, valid_ys = zip(*[(x.T, y) for x, y in valid_data])

D, K = gamma.shape

for xs, ys in [(x.T, y) for x, y in valid_data]:
    pred = crf.posteriorMAP(xs, theta, gamma, D)
    print 'pred:  ', [int(k) for k, v in list(groupby(pred)) if int(k) > 0]
    print 'actual:', [int(k) for k, v in list(groupby(ys)) if int(k) > 0]
    print
