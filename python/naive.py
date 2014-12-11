from glob import glob
import random
from gatherData import gatherRandomXY, TRAIN_FILE_PATTERN, VALID_FILE_PATTERN
import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import matplotlib.pyplot as plt

import multiprocessing as mp
import os

train_num_files = 200
samples_per_file = 250
train_data_files = random.sample(glob(TRAIN_FILE_PATTERN), train_num_files)

validation_num_files = 200
# validation_samples_per_file = 250
validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)

# One-sided length of window. Total window length will be twice this value.
window_size = 15

print '... Gathering data'

def _gather(fn):
    return gatherRandomXY(fn, window_size, samples_per_file, onlyGesture=False)

pool = mp.Pool()
train_data = pool.map(_gather, train_data_files)
Xtrain = np.vstack([x for (xs, ys) in train_data for x in xs])
Ytrain = np.array([y for (xs, ys) in train_data for y in ys])
del train_data

validation_data = pool.map(_gather, validation_data_files)
Xvalid = np.vstack([x for (xs, ys) in validation_data for x in xs])
Yvalid = np.array([y for (xs, ys) in validation_data for y in ys])
del validation_data

print Xtrain.shape, len(Ytrain)

# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)
# del X, Y

results_dir = 'results/naive_train_%d_validation_%d_windowsize_%d/' % (train_num_files, validation_num_files, window_size)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def testClassifier(clf):
    clf = clf.fit(Xtrain, Ytrain)
    cm = confusion_matrix(Yvalid, clf.predict(Xvalid))
    return clf, clf.score(Xvalid, Yvalid), cm

def showConfusionMatrix(cm, title, show=True, outfile=None):
    plt.figure()
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
# showConfusionMatrix(svm_cm, 'RBF SVM', outfile='results/svm_confusion_matrix.png')

print '... Training Logistic Regression'
# lr_clf = LogisticRegression(penalty='l2').fit(Xtrain, Ytrain)
# print lr_clf.score(Xvalid, Yvalid)
lr_clf, lr_score, lr_cm = testClassifier(LogisticRegression(penalty='l2'))
# lr_clf, lr_score, lr_cm = testClassifier(SGDClassifier(loss='log',
#                                                        penalty='l1',
#                                                        shuffle=True,
#                                                        n_jobs=-1,
#                                                        verbose=2))
print lr_score
showConfusionMatrix(lr_cm, 'Logistic Regression', show=False, outfile=results_dir + 'logistic.png')

# print '... Training Extra Trees Classifier'
# et_clf, et_score, et_cm = testClassifier(ExtraTreesClassifier(n_estimators=25, n_jobs=1))
# print et_score
# showConfusionMatrix(et_cm, 'Extra Trees Classifier', outfile='et_confusion_matrix.png')

print '... Training Random Forest'
rf_clf, rf_score, rf_cm = testClassifier(RandomForestClassifier(n_estimators=250, n_jobs=-1))
print rf_score
showConfusionMatrix(rf_cm, 'Random Forest', show=False, outfile=results_dir + 'randomforest.png')
