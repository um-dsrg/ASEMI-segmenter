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

'''
Module for hash functions for representing images with small numeric vectors.

In order to find which volume slice images correspond to which subvolume slice images, we need to
be able to perform the search in memory. It is not practical to do so with full images and so a
small hash is needed in order to be able to compare the hashes instead of the full images. These
hash vectors would fit in memory and also be compatible with distance functions to find the nearest
neighbour rather than exact matches as pixels might change slightly when saved in different
formats.
'''

import numpy as np
import os
import sys
from asemi_segmenter.lib import validations


#########################################
def load_hashfunction_from_config(config):
    '''
    Load a hash function from a configuration dictionary.

    :param dict config: Configuration of the hash function.
    :return: A hash function object.
    :rtype: HashFunction
    '''
    validations.validate_json_with_schema_file(config, 'hash_function.json')

    if config['type'] == 'random_indexing':
        hash_size = config['params']['hash_size']
        return RandomIndexingHashFunction(hash_size)


#########################################
class HashFunction(object):
    '''Super class for hash functions.'''

    #########################################
    def __init__(self, hash_size, name):
        '''
        Constructor.

        :param int hash_size: The size of the hash vector.
        :param str name: The name of the hash function.
        '''
        self.name = name
        self.hash_size = hash_size

    #########################################
    def init(self, input_shape, seed=None):
        '''
        Initialise the hash function's data.

        :param tuple input_shape: The shape of the input array to hash.
        :param int seed: The seed of any random functions used.
        '''
        raise NotImplementedError()

    #########################################
    def apply(self, data):
        '''
        Get the hash vector of an input array.

        :param nump.ndarray data: The input array.
        :return: The hash vector.
        :rtype: numpy.ndarray
        '''
        raise NotImplementedError()


#########################################
class RandomIndexingHashFunction(HashFunction):
    '''Random indexing (https://en.wikipedia.org/wiki/Random_indexing).'''

    #########################################
    def __init__(self, hash_size, name='random_indexing'):
        '''
        Constructor.

        :param int hash_size: The size of the hash vector.
        :param str name: The name of the hash function.
        '''
        super().__init__(hash_size, name)
        self.indexer = None

    #########################################
    def init(self, input_shape, seed=None):
        '''
        Initialise the hash function's data.

        :param tuple input_shape: The shape of the input array to hash.
        :param int seed: The seed of the random matrix.
        '''
        rand = np.random.RandomState(seed)
        tmp = 2*rand.random_sample(size=[ np.prod(input_shape), self.hash_size ]) - 1
        self.indexer = (
                np.where(tmp < -0.95, np.array(-1, np.int8), np.array(0, np.int8)) +
                np.where(tmp > 0.95, np.array(1, np.int8), np.array(0, np.int8))
            )

    #########################################
    def apply(self, data):
        '''
        Get the hash vector of an input array.

        :param nump.ndarray data: The input array.
        :return: The hash vector.
        :rtype: numpy.ndarray
        '''
        return np.dot(data.reshape([-1]), self.indexer).astype(np.float32)
