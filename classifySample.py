from glob import glob
import random
from gatherData import gatherAllXY, gatherRandomXY
import numpy as np
from scipy import stats

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import matplotlib.pyplot as plt
import multiprocessing as mp

def predict(clf, fn):
    X, Y = gatherAllXY(fn, window_size)
    prediction = clf.predict(X)
    return smooth(prediction)

def smooth(pred):
    frame_rate = 20
    smoothed = np.zeros(pred.shape)
    for i in range(len(shape)):
        neighbors = np.array([pred(x) for x in range(min(i-frame_rate/2, 0), max(i+frame_rate/2,0))])
        smoothed[i] = stats.mode(neighbors)[0][0]
    return smoothed


train_num_files = 200
samples_per_file = 250
train_file_pattern = '/users/skainswo/data/skainswo/chalearn/train/Sample*_data.mat'
all_train_files = glob(train_file_pattern)
test_file = all_train_files[0]
train_data_files = random.sample(all_train_files[1:], train_num_files)

# validation_num_files = 200
# validation_file_pattern = '/users/skainswo/data/skainswo/chalearn/validation/Sample*_data.mat'
# validation_data_files = random.sample(glob(validation_file_pattern), validation_num_files)


# One-sided length of window. Total window length will be twice this value.
window_size = 15

print '... Gathering data'

def _gatherFile(fn):
    return gatherRandomXY(fn, window_size, samples_per_file)

pool = mp.Pool()
train_data = pool.map(_gatherFile, train_data_files)
Xtrain = np.vstack([x for (xs, ys) in train_data for x in xs])
Ytrain = np.array([y for (xs, ys) in train_data for y in ys])
del train_data

print Xtrain.shape, len(Ytrain)

def trainAndPredict(clf, fn):
    clf = clf.fit(Xtrain, Ytrain)
    pred = predict(clf,fn)
    return pred

print 'training and predicting'
print trainAndPredict(RandomForestClassifier(n_estimators=250, n_jobs=-1), test_file)
