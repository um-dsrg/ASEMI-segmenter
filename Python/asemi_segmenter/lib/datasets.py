#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.
#
# ASEMI-segmenter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ASEMI-segmenter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASEMI-segmenter.  If not, see <http://www.gnu.org/licenses/>.

'''Module for data set functions.'''

import random
import h5py
import numpy as np
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import featurisers


#########################################
class DataSet(object):
    '''Data set of voxel features to voxel labels.'''

    #########################################
    def __init__(self, data_fullfname):
        '''
        Constructor.

        :param data_fullfname: The full file name (with path) to the HDF file if to be used or
            None if data set will be a numpy array kept in memory.
        :type data_fullfname: str or None
        '''
        self.data_fullfname = data_fullfname
        self.data = None

    #########################################
    def create(self, num_items, feature_size):
        '''
        Create the HDF file or numpy array.

        :param int num_items: The number of voxels in the data set.
        :param int feature_size: The number of elements in the feature vectors describing
            the voxels.
        '''
        if self.data_fullfname is not None:
            with h5py.File(self.data_fullfname, 'w') as data_f:
                data_f.create_dataset('labels', [num_items], dtype=np.uint8, chunks=None)
                data_f.create_dataset(
                    'features',
                    [num_items, feature_size],
                    dtype=featurisers.feature_dtype,
                    chunks=None
                    )
        else:
            self.data = {
                'labels': np.empty([num_items], dtype=np.uint8),
                'features': np.empty([num_items, feature_size], dtype=featurisers.feature_dtype)
                }

    #########################################
    def load(self, as_readonly=False):
        '''Load an existing HDF file using the file path given in the constructor.'''
        if self.data_fullfname is not None:
            self.data = h5py.File(self.data_fullfname, 'r' if as_readonly else 'r+')

    #########################################
    def get_labels_array(self):
        '''
        Get the labels column of the data set.

        :return: An array of labels.
        :rtype: h5py.Dataset or numpy.ndarray
        '''
        return self.data['labels']

    #########################################
    def get_features_array(self):
        '''
        Get the features column of the data set.

        :return: A 2D array of features.
        :rtype: h5py.Dataset or numpy.ndarray
        '''
        return self.data['features']

    #########################################
    def without_control_labels(self):
        '''
        Get a copy of this data set without any items where the labels are control labels.
        '''
        valid_items_mask = self.data['labels'][:] < volumes.FIRST_CONTROL_LABEL

        new_dataset = DataSet(None)
        new_dataset.create(np.sum(valid_items_mask), self.data['features'].shape[1])

        block_size = 100000
        j = 0
        for i in range(0, valid_items_mask.shape[0], block_size):
            blocked_mask = valid_items_mask[i:i+block_size]
            block_out_size = np.sum(blocked_mask)
            new_dataset.get_labels_array()[j:j+block_out_size] = self.data['labels'][i:i+block_size][blocked_mask]
            new_dataset.get_features_array()[j:j+block_out_size] = self.data['features'][i:i+block_size, :][blocked_mask, :]
            j += block_out_size

        return new_dataset

    #########################################
    def close(self):
        '''Close the HDF file (if used and open).'''
        if self.data is not None:
            if self.data_fullfname is not None:
                self.data.close()
            self.data = None


#########################################
def sample_voxels(loaded_labels, max_sample_size_per_label, num_labels, volume_slice_indexes_in_subvolume, slice_shape, skip=0, seed=None):
    '''
    Get a balanced random sample of voxel indexes.

    Sample is balanced among labels provided that there are enough
    of each label (otherwise all the items of a label will be returned).

    :param numpy.ndarray loaded_labels: 1D numpy array of label indexes
        from a number of full slices.
    :param int max_sample_size_per_label: The number of items from each label to
        return in the new data set. If there are less items than this then all the items
        are returned.
    :param int num_labels: The number of labels to consider such that the last
        label index is num_labels-1.
    :param list volume_slice_indexes_in_subvolume: The volume indexes of all the slices
        in loaded_labels in order of appearance in loaded_labels.
    :param tuple slice_shape: Tuple with the numpy shape of each slice.
    :param int skip: The number of voxels to skip before selecting. This is used
        for when the same slices are used for separate datasets and you want
        the second dataset to avoid the voxels that were selected for the first.
    :param int seed: The random number generator seed to use when randomly selecting data set
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
            subvolume_slice = pos//slice_size
            slc = volume_slice_indexes_in_subvolume[subvolume_slice]
            pos -= subvolume_slice*slice_size
            row = pos//num_cols
            pos -= row*num_cols
            col = pos
            positions_result.append((slc, row, col))
        labels_result.append(slice(label_segment_start, label_segment_start+len(label_positions)))
        label_segment_start += len(label_positions)
    return (positions_result, labels_result)
