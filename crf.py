from glob import glob
import random
from gatherData import gatherAllXY, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN
import numpy as np

import multiprocessing as mp


train_num_files = 200
train_file_pattern = '/users/skainswo/data/skainswo/chalearn/train/Sample*_data.mat'
train_data_files = random.sample(glob(train_file_pattern), train_num_files)

validation_num_files = 200
validation_file_pattern = '/users/skainswo/data/skainswo/chalearn/validation/Sample*_data.mat'
validation_data_files = random.sample(glob(validation_file_pattern), validation_num_files)


# One-sided length of window. Total window length will be twice this value.
window_size = 15

print '... Gathering data'

pool = mp.Pool()
train_data = pool.map(lambda fn: gatherAllXY(fn, window_size), train_data_files)
Xtrain = np.vstack([x for (xs, ys) in train_data for x in xs])
Ytrain = np.vstack([y for (xs, ys) in train_data for y in ys])
del train_data

validation_data = pool.map(lambda fn: gatherAllXY(fn, window_size), validation_data_files)
Xvalid = np.vstack([x for (xs, ys) in validation_data for x in xs])
Yvalid = np.vstack([y for (xs, ys) in validation_data for y in ys])
del validation_data

print Xtrain.shape, len(Ytrain)
