import numpy as np
import os
import sys

#########################################
class HashFunction(object):
    
    #########################################
    def __init__(self, hash_size, name):
        self.name = name
        self.hash_size = hash_size
        
    #########################################
    def init(self, input_shape, seed=None):
        raise NotImplementedError()
    
    #########################################
    def apply(self, data):
        raise NotImplementedError()

#########################################
class RandomIndexingHashFunction(HashFunction):
    
    #########################################
    def __init__(self, hash_size, name='random_indexing'):
        super().__init__(hash_size, name)
        self.indexer = None
    
    #########################################
    def init(self, input_shape, seed=None):
        rand = np.random.RandomState(seed)
        tmp = 2*rand.random(size=[ np.prod(input_shape), self.hash_size ]) - 1
        self.indexer = (
                np.where(tmp < -0.95, np.array(-1, np.int8), np.array(0, np.int8)) +
                np.where(tmp > 0.95, np.array(1, np.int8), np.array(0, np.int8))
            )
    
    #########################################
    def apply(self, data):
        return np.dot(data.reshape([-1]), self.indexer).astype(np.float32)