import CRF_learning as crf
from glob import glob
import random
from gatherData import allLabels, gatherAllXYNoWindow, gatherAllXY, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN

# import numpy as np
# import matplotlib.pyplot as plt


train_num_files = 50
train_data_files = random.sample(glob(TRAIN_FILE_PATTERN), train_num_files)

validation_num_files = 5
validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)


print '... Gathering data'

window_size = 5
def _gather(fn):
    return gatherAllXY(fn, 5)

pool = mp.Pool()
train_data = pool.map(_gather, train_data_files)
valid_data = pool.map(_gather, validation_data_files)

train_xs, train_ys = zip(*train_data)
valid_xs, valid_ys = zip(*valid_data)

_, K = train_xs[0].shape
D = len(allLabels)

crf.learnParameters(train_xs, train_ys, K, D)
