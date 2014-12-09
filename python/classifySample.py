from glob import glob
import random
from gatherData import gatherAllXY, gatherRandomXY, loadFile
import numpy as np
from scipy import stats
from itertools import groupby

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
    for i in range(smoothed.size):
        min_index = max(i-frame_rate/2, 0)
        max_index = min(i+frame_rate/2,smoothed.size)
        neighbors = np.array([pred[x] for x in range(min_index, max_index)])
        smoothed[i] = stats.mode(neighbors)[0][0]
    return [k for k, g in groupby(smoothed)]


train_num_files = 100
samples_per_file = 200
train_file_pattern = '/users/skainswo/data/skainswo/chalearn/train/Sample*_data.mat'
all_train_files = glob(train_file_pattern)
#test_file = all_train_files[0]
#print "test file", test_file
train_data_files = random.sample(all_train_files, train_num_files)

validation_num_files = 50
validation_file_pattern = '/users/skainswo/data/skainswo/chalearn/validation/Sample*_data.mat'
all_validation_files = glob(validation_file_pattern)
validation_data_files = random.sample(all_validation_files, validation_num_files)

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

def trainClassifier(clf):
    clf = clf.fit(Xtrain, Ytrain)
    return clf

#def trainAndPredict(clf, fn):
#    clf = clf.fit(Xtrain, Ytrain)
#    pred = predict(clf,fn)
#    return pred

print '...training'
clf = trainClassifier(RandomForestClassifier(n_estimators=250, n_jobs=-1))

#copied from http://hetland.org/coding/python/levenshtein.py
def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]

print '...done training'
lev_dists = []
for valid_file in validation_data_files:
    pred = predict(clf, valid_file)
    actual = [k for k, g in groupby(loadFile(valid_file)['frame_labels']) if k > 0]
    print levenshtein(pred,actual)
    print 'length of actual: ', actual
    lev_dists = lev_dists + [levenshtein(pred,actual)]

#print 'predicted', trainAndPredict(RandomForestClassifier(n_estimators=250, n_jobs=-1), test_file)
#print 'actual', [k for k, g in groupby(loadFile(test_file)['frame_labels']) if k > 0]

print lev_dists
