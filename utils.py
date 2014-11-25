import scipy.io
import numpy as np

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
