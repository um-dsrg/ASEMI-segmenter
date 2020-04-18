'''Module for training set functions.'''

import random
import h5py
import numpy as np
from asemi_segmenter.lib import volumes


#########################################
class TrainingSet(object):
    '''Training set of voxel features to voxel labels.'''

    #########################################
    def __init__(self, data_fullfname):
        '''
        Create a training set object.

        :param data_fullfname: The full file name (with path) to the HDF file if to be used or
            None if training set will be a numpy array kept in memory.
        :type data_fullfname: str or None
        '''
        self.data_fullfname = data_fullfname
        self.data = None

    #########################################
    def create(self, num_items, feature_size):
        '''
        Create the HDF file or numpy array.

        :param int num_items: The number of voxels in the training set.
        :param int feature_size: The number of elements in the feature vectors describing
            the voxels.
        '''
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                data_f.create_dataset('labels', [num_items], dtype=np.uint8, chunks=None)
                data_f.create_dataset(
                    'features',
                    [num_items, feature_size],
                    dtype=np.float32,
                    chunks=None
                    )
        else:
            self.data = {
                'labels': np.empty([num_items], dtype=np.uint8),
                'features': np.empty([num_items, feature_size], dtype=np.float32)
                }

    #########################################
    def load(self):
        '''Load the HDF file (if data_fullfname was not None).'''
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r+')

    #########################################
    def get_labels_array(self):
        '''
        Get the labels column of the training set.

        :return: An array of labels.
        :rtype: h5py.Dataset or numpy.ndarray
        '''
        return self.data['labels']

    #########################################
    def get_features_array(self):
        '''
        Get the features column of the training set.

        :return: A 2D array of features.
        :rtype: h5py.Dataset or numpy.ndarray
        '''
        return self.data['features']

    #########################################
    def without_control_labels(self):
        '''
        Get a copy of this training set without any items where the labels are control labels.
        '''
        valid_items_mask = self.data['labels'][:] < volumes.FIRST_CONTROL_LABEL
        
        new_trainingset = TrainingSet(None)
        new_trainingset.create(np.sum(valid_items_mask), self.data['features'].shape[1])
        new_trainingset.get_labels_array()[:] = self.data['labels'][valid_items_mask]
        new_trainingset.get_features_array()[:] = self.data['features'][valid_items_mask, :]

        return new_trainingset

    #########################################
    def close(self):
        '''Close the HDF file (if used and open).'''
        if self.data is not None:
            self.data.close()
            self.data = None
            

#########################################
def sample_voxels(loaded_labels, max_sample_size_per_label, num_labels, slice_shape, skip=0, seed=None):
    '''
    Get a balanced random sample of voxel indexes.

    Sample is balanced among labels provided that there are enough
    of each label (otherwise all the items of a label will be returned).

    :param numpy.ndarray loaded_labels: 1D numpy array of label indexes
        from a number of full slices.
    :param int max_sample_size_per_label: The number of items from each label to
        return in the new training set. If there are less items than this then all the items
        are returned.
    :param int num_labels: The number of labels to consider such that the last
        label index is num_labels-1.
    :param tuple slice_shape: Tuple with the numpy shape of each slice.
    :param int skip: The number of voxels to skip before selecting. This is used
        for when the same slices are used for separate datasets and you want
        the second dataset to avoid the voxels that were selected for the first.
    :param int seed: The random number generator seed to use when randomly selecting training
        items.
    :return A tuple consisting of (indexes, labels). 'indexes' is a
        list of voxel indexes sorted by corresponding label index. Each
        index is a tuple consisting of (slice, row, column) indexes of a
        given voxel. 'labels' is a list of Python slices such that
        indexes[labels[i]] gives all the indexes of the ith label.
    :rtype: tuple
    '''
    (num_rows, num_cols) = slice_shape
    slice_size = num_rows*num_cols
    num_slcs = loaded_labels.size//slice_size
    
    all_positions = np.arange(loaded_labels.size)
    r = random.Random(seed)
    positions_result = list()
    labels_result = list()
    label_segment_start = 0
    for label_index in range(num_labels):
        label_positions = all_positions[loaded_labels == label_index].tolist()
        r.shuffle(label_positions)
        label_positions = label_positions[skip:skip+max_sample_size_per_label]
        for pos in label_positions:
            slc = pos//slice_size
            pos -= slc*slice_size
            row = pos//num_cols
            pos -= row*num_cols
            col = pos
            positions_result.append((slc, row, col))
        labels_result.append(slice(label_segment_start, label_segment_start+len(label_positions)))
        label_segment_start += len(label_positions)
    return (positions_result, labels_result)