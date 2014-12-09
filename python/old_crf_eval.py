import random
from glob import glob
from gatherData import gatherAllXYNoWindow, VALID_FILE_PATTERN
import pycrfsuite
from itertools import groupby

import multiprocessing as mp

validation_num_files = 5
validation_data_files = random.sample(glob(VALID_FILE_PATTERN), validation_num_files)

print '... Gathering data'

pool = mp.Pool()
validation_data = pool.map(gatherAllXYNoWindow, validation_data_files)

tagger = pycrfsuite.Tagger()
tagger.open('crfsuite_basic_crf.model')
for xseq, yseq in validation_data:
    print 'pred:  ', [int(k) for k, v in list(groupby(tagger.tag(xseq)))]
    print 'actual:', [int(k) for k, v in list(groupby(yseq)) if int(k) > 0]
    print
