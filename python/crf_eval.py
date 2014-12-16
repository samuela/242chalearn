import custom_crf_base as crf

import random
from glob import glob
from gatherData import gatherAllXY, VALID_FILE_PATTERN
from itertools import groupby
import numpy as np

from sklearn.metrics import confusion_matrix

from scipy.io import savemat

# import multiprocessing as mp

import matplotlib.pyplot as plt

# validation_num_files = 1
# validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)
validation_data_files = glob(VALID_FILE_PATTERN)

print '... Gathering data'

theta = np.load('results/crf_train_393_windowsize_15_lamb_1.000000_maxiter_500/theta.npy')
gamma = np.load('results/crf_train_393_windowsize_15_lamb_1.000000_maxiter_500/gamma.npy')

window_size = 15
def _gather(fn):
    return gatherAllXY(fn, window_size)

# pool = mp.Pool()
# valid_data = pool.map(_gather, validation_data_files)
# valid_xs, valid_ys = zip(*[(x.T, y) for x, y in valid_data])

D, K = gamma.shape

# theta = np.zeros((D, D))

# for xs, ys in [(x.T, y) for x, y in valid_data]:

shit = {}

cm = np.zeros((D, D))

for fn in validation_data_files:
    k = fn.replace('/users/skainswo/data/skainswo/chalearn/validation/', '').replace('_data.mat', '')

    xs, ys = gatherAllXY(fn, window_size)
    xs = xs.T
    pred = crf.posteriorMAP(xs, theta, gamma, D)
    samples = crf.samplePosterior(xs, theta, gamma, D, 1)
    # print 'pred:  ', [int(k) for k, v in list(groupby(pred)) if int(k) > 0]
    # print 'actual:', [int(k) for k, v in list(groupby(ys)) if int(k) > 0]
    # print

    per_frame = np.vstack([samples, pred, ys])
    shit[k] = per_frame

    cm += confusion_matrix(ys, pred, labels=range(D))
    
    # plt.figure(num=1, figsize=(12, 4))
    # plt.matshow(per_frame[:,per_frame[2,:] > 0], fignum=1, aspect=per_frame.shape[1] / 50, cmap='gist_stern')
    # plt.show()

# savemat('results/shitfojake_no_theta.mat', shit)

np.save('results/crf_train_393_windowsize_15_lamb_1.000000_maxiter_500/conf_mat.npy', cm)

plt.matshow(cm[1:,1:])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CRF without theta parameters')
plt.savefig('results/crf_train_393_windowsize_15_lamb_1.000000_maxiter_500_randominit/conf_mat_no_theta.png')

# xs, ys = gatherAllXY(fn, window_size)
# xs = xs.T
# pred = crf.posteriorMAP(xs, theta, gamma, D)
# print 'pred:  ', [int(k) for k, v in list(groupby(pred)) if int(k) > 0]
# print 'actual:', [int(k) for k, v in list(groupby(ys)) if int(k) > 0]
# print

# plt.matshow([pred, ys])
# plt.show()
