import scipy.io
import numpy as np
import random

allLabels = ['*NONE*',
             'vattene',
             'vieniqui',
             'perfetto',
             'furbo',
             'cheduepalle',
             'chevuoi',
             'daccordo',
             'seipazzo',
             'combinato',
             'freganiente',
             'ok',
             'cosatifarei',
             'basta',
             'prendere',
             'noncenepiu',
             'fame',
             'tantotempo',
             'buonissimo',
             'messidaccordo',
             'sonostufo']

numForLabel = {l: i for (i, l) in enumerate(allLabels)}

jointTypes = ['HipCenter',
              'Spine',
              'ShoulderCenter',
              'Head',
              'ShoulderLeft',
              'ElbowLeft',
              'WristLeft',
              'HandLeft',
              'ShoulderRight',
              'ElbowRight',
              'WristRight',
              'HandRight',
              'HipLeft',
              'KneeLeft',
              'AnkleLeft',
              'FootLeft',
              'HipRight',
              'KneeRight',
              'AnkleRight',
              'FootRight']

TRAIN_FILE_PATTERN = '/users/skainswo/data/skainswo/chalearn/train/Sample*_data.mat'
VALID_FILE_PATTERN = '/users/skainswo/data/skainswo/chalearn/validation/Sample*_data.mat'

numForJoint = {l: i for (i, l) in enumerate(jointTypes)}

def loadFile(fn):
    mat = scipy.io.loadmat(fn)

    # Lots of [0]'s since loading mat's is messy
    num_frames = mat['Video']['NumFrames'][0][0][0][0]
    frame_rate = mat['Video']['FrameRate'][0][0][0][0]

    wp = lambda i: mat['Video']['Frames'][0][0]['Skeleton'][0][i]['WorldPosition'][0][0]
    wr = lambda i: mat['Video']['Frames'][0][0]['Skeleton'][0][i]['WorldRotation'][0][0]
    m_wp, n_wp = wp(0).shape
    m_wr, n_wr = wr(0).shape

    world_positions = np.zeros((num_frames, m_wp, n_wp))
    world_rotations = np.zeros((num_frames, m_wr, n_wr))

    for i in xrange(num_frames):
        world_positions[i,:,:] = wp(i)
        world_rotations[i,:,:] = wr(i)

    # Convert to dense format. Not all frames are associated with a label.
    # Otherwise unlabeled frames are assigned the 0 label.
    label_pos = mat['Video']['Labels'][0][0][0]
    frame_labels = np.zeros((num_frames,))
    for label, start, end in label_pos:
        frame_labels[start[0][0] - 1:end[0][0]] = numForLabel[label[0]]

    # print label_pos
    # print frame_labels

    # print num_frames, frame_rate, world_positions.shape, world_rotations.shape
    # print frame_labels
    # print

    return {'world_position': world_positions,
            'world_rotation': world_rotations,
            'frame_labels': frame_labels,
            'num_frames': num_frames,
            'frame_rate': frame_rate}

def centerToHip(pos):
    return pos - pos[:,0,:].reshape(pos.shape[0], 1, pos.shape[2])

def gatherRandomXY(fn, window_size, samples_per_file, onlyGesture=False):
    data = loadFile(fn)
#    pos = data['world_position']
    pos = centerToHip(data['world_position'])
    rot = data['world_rotation']
    num_frames = data['num_frames']
    labels = data['frame_labels']

    # Select indices uniformly at random.
    # all_ixs = xrange(window_size, num_frames - window_size)

    # Select only those indices corresponding to gestures.
    if onlyGesture:
        all_ixs = window_size + np.where(labels[window_size:num_frames - window_size] > 0)[0]
    else:
        all_ixs = window_size + np.where(labels[window_size:num_frames - window_size] >= 0)[0]

    X, Y = [], []
    for ix in random.sample(all_ixs, min(samples_per_file, len(all_ixs))):
        X.append(np.hstack((pos[ix - window_size:ix + window_size].ravel(),
                            rot[ix - window_size:ix + window_size].ravel())))
        Y.append(labels[ix])

    return X, Y

def gatherAllXY(fn, window_size, diff=False):
    X, Y = gatherAllXYNoWindow(fn)

    if diff:
        X = np.diff(X, axis=0)
        Y = Y[1:]

    shape = (X.shape[0] - window_size*2 + 1, X.shape[1] * window_size * 2)
    x = np.lib.stride_tricks.as_strided(X, shape=shape, strides=X.strides)
    y = Y[window_size - 1:Y.size - window_size]

#    print x.shape, len(y)

    return x, y

def gatherAllXYNoWindow(fn):
    data = loadFile(fn)
    pos = centerToHip(data['world_position'])
    rot = data['world_rotation']
    labels = data['frame_labels']

    ps, rs = pos.shape, rot.shape

    return np.hstack((pos.reshape(ps[0], ps[1] * ps[2]),
                      rot.reshape(rs[0], rs[1] * rs[2]))), labels
