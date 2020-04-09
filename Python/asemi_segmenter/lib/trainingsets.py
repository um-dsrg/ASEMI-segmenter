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
    def get_sample(self, max_sample_size_per_label, seed=None):
        '''
        Get a random sample of the training set.

        Sample is always kept in memory and the training set will be balanced among labels
        provided that there are enough of each label (otherwise all the items of a label
        will be returned).

        :param int max_sample_size_per_label: The number of items from each label to
            return in the new training set. If there are less items than this then all the items
            are returned.
        :param int seed: The random number generator seed to use when randomly selecting training
            items.
        :return The sub training set.
        :rtype: TrainingSet
        '''
        label_locations = dict()
        for (i, label) in enumerate(self.data['labels'][:].tolist()):
            if label < volumes.FIRST_CONTROL_LABEL:
                if label not in label_locations:
                    label_locations[label] = list()
                label_locations[label].append(i)
        num_labels = len(label_locations)

        for label in range(num_labels):
            r = random.Random(seed)
            r.shuffle(label_locations[label])

        all_locations = [
            location
            for label in range(num_labels)
            for location in label_locations[label][:max_sample_size_per_label]]
        all_locations.sort()
        total_items_samples = len(all_locations)

        new_trainingset = TrainingSet(None)
        new_trainingset.create(total_items_samples, self.data['features'].shape[1])
        new_trainingset.get_labels_array()[:] = self.data['labels'][all_locations]
        new_trainingset.get_features_array()[:] = self.data['features'][all_locations, :]

        return new_trainingset

    #########################################
    def close(self):
        '''Close the HDF file (if used and open).'''
        if self.data is not None:
            self.data.close()
            self.data = None