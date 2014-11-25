from glob import glob
import random
from utils import loadFile
import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import matplotlib.pyplot as plt

import multiprocessing as mp


train_num_files = 200
samples_per_file = 250
train_file_pattern = '/users/skainswo/data/skainswo/chalearn/train/Sample*_data.mat'
train_data_files = random.sample(glob(train_file_pattern), train_num_files)

validation_num_files = 200
# validation_samples_per_file = 250
validation_file_pattern = '/users/skainswo/data/skainswo/chalearn/validation/Sample*_data.mat'
validation_data_files = random.sample(glob(validation_file_pattern), validation_num_files)


# One-sided length of window. Total window length will be twice this value.
window_size = 15

print '... Gathering data'

# Xtrain = []
# Ytrain = []

def gatherXY(fn):
    data = loadFile(fn)
    pos = data['world_position']
    rot = data['world_rotation']
    num_frames = data['num_frames']
    labels = data['frame_labels']

    # Select indices at uniformly at random.
    # all_ixs = xrange(window_size, num_frames - window_size)

    # Select only those indices corresponding to gestures.
    all_ixs = window_size + np.where(labels[window_size:num_frames - window_size] > 0)[0]

    X, Y = [], []
    for ix in random.sample(all_ixs, min(samples_per_file, len(all_ixs))):
        X.append(np.hstack((pos[ix - window_size:ix + window_size].ravel(),
                            rot[ix - window_size:ix + window_size].ravel())))
        Y.append(labels[ix])

    return X, Y


# for fn in data_files:
#     data = loadFile(fn)
#     pos = data['world_position']
#     rot = data['world_rotation']
#     num_frames = data['num_frames']
#     labels = data['frame_labels']

    # Select indices at uniformly at random.
    # all_ixs = xrange(window_size, num_frames - window_size)

    # Select only those indices corresponding to gestures.
#     all_ixs = window_size + np.where(labels[window_size:num_frames - window_size] > 0)[0]

#     for ix in random.sample(all_ixs, min(samples_per_file, len(all_ixs))):
#         Xtrain.append(np.hstack((pos[ix - window_size:ix + window_size].ravel(),
#                                  rot[ix - window_size:ix + window_size].ravel())))
#         Ytrain.append(labels[ix])

# X = np.vstack(X)
# train_data = pool.map(lambda fn: gatherXY(fn, window_size, train_samples_per_file),
#                       train_data_files)

pool = mp.Pool()
train_data = pool.map(gatherXY, train_data_files)
Xtrain = np.vstack([x for (xs, ys) in train_data for x in xs])
Ytrain = np.vstack([y for (xs, ys) in train_data for y in ys])
del train_data

validation_data = pool.map(gatherXY, validation_data_files)
Xvalid = np.vstack([x for (xs, ys) in validation_data for x in xs])
Yvalid = np.vstack([y for (xs, ys) in validation_data for y in ys])
del validation_data

print Xtrain.shape, len(Ytrain)

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)
# del X, Y

def testClassifier(clf):
    clf = clf.fit(Xtrain, Ytrain)
    cm = confusion_matrix(Yvalid, clf.predict(Xvalid))
    return clf, clf.score(Xvalid, Yvalid), cm

def showConfusionMatrix(cm, title, show=True, outfile=None):
    # plt.figure()
    plt.matshow(cm)
    plt.title(title)
    # plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()


# print '... Training SVM'
# svm_clf, svm_score, svm_cm = testClassifier(svm.SVC())
# print svm_score
# showConfusionMatrix(svm_cm, 'RBF SVM', outfile='svm_confusion_matrix.png')

# print '... Training Logistic Regression'
# lr_clf = LogisticRegression(penalty='l1').fit(Xtrain, Ytrain)
# print lr_clf.score(Xtest, Ytest)
# lr_clf, lr_score, lr_cm = testClassifier(LogisticRegression(penalty='l1'))
# lr_clf, lr_score, lr_cm = testClassifier(SGDClassifier(loss='log',
#                                                        penalty='l1',
#                                                        shuffle=True,
#                                                        n_jobs=-1,
#                                                        verbose=2))
# print lr_score
# showConfusionMatrix(lr_cm, 'Logistic Regression', show=False, outfile='lr_confusion_matrix.png')

# print '... Training Extra Trees Classifier'
# et_clf, et_score, et_cm = testClassifier(ExtraTreesClassifier(n_estimators=25, n_jobs=1))
# print et_score
# showConfusionMatrix(et_cm, 'Extra Trees Classifier', outfile='et_confusion_matrix.png')

print '... Training Random Forest'
rf_clf, rf_score, rf_cm = testClassifier(RandomForestClassifier(n_estimators=250, n_jobs=-1))
print rf_score
showConfusionMatrix(rf_cm, 'Random Forest', outfile='rf_confusion_matrix.png')
