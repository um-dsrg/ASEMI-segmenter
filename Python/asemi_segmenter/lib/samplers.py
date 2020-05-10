'''Random generators for the purpose of hyperparameter random search.'''

import random
import math

#########################################
class Sampler(object):
    '''Super class for samplers.'''
    
    #########################################
    def __init__(self, seed):
        '''
        Constructor.
        '''
        self.seed = seed
        self.initialised = False
        self.value = None
    
    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        raise NotImplementedError()
    
    #########################################
    def get_value(self):
        '''
        Get the random value that was generated.
        
        :return: The random value.
        :rtype: generic
        '''
        if not self.initialised:
            raise RuntimeError('Sampler value was not set (call resample method).')
        return self.value


#########################################
class ConstantSampler(Sampler):
    '''Fixed value sampler (doesn't change).'''
    
    #########################################
    def __init__(self, value, seed=None):
        '''
        Constructor.
        
        :param generic value: Value to return.
        :param object seed: Seed to the random number generator,
            if there was one (does nothing).
        '''
        super().__init__(seed)
        
        self.value = value
    
    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.initialised = True


#########################################
class IntegerSampler(Sampler):
    '''Sample a random integer.'''
    
    #########################################
    def __init__(self, min, max, distribution, seed=None):
        '''
        Constructor.
        
        :param int min: Minimum value (inclusive).
        :param int max: Maximum value (inclusive).
        :param string distribution: One of the following values:
            'uniform': All values between min and max are
            equally likely to be sampled.
            'log2': Only powers of 2 between min and max will be sampled.
        :param object seed: Seed to the random number generator.
        '''
        super().__init__(seed)
        
        if min > max:
            raise ValueError('min cannot be greater than max (min={}, max={}).'.format(min, max))
        if distribution not in ['uniform', 'log2', 'log10']:
            raise ValueError('distribution must be uniform or log2, not {}.'.format(distribution))
        if distribution == 'log2':
            if min <= 0:
                raise ValueError('When distribution is logarithmic, min must be a positive number, not {}.'.format(min))
            if max <= 0:
                raise ValueError('When distribution is logarithmic, max must be a positive number, not {}.'.format(max))
        if distribution == 'log2':
            if math.log2(min)%1 != 0:
                raise ValueError('When distribution is log2, min must be a power of 2, not {}.'.format(min))
            if math.log2(max)%1 != 0:
                raise ValueError('When distribution is log2, max must be a power of 2, not {}.'.format(max))
        
        self.min = min
        self.max = max
        self.distribution = distribution
        self.rng = random.Random(seed)
        self.method = {
            'uniform': lambda:self.rng.randint(self.min, self.max),
            'log2':    lambda:2**self.rng.randint(math.log2(self.min), math.log2(self.max))
            }[distribution]
    
    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.value = self.method()
        self.initialised = True


#########################################
class FloatSampler(Sampler):
    '''Sample a random float.'''
    
    #########################################
    def __init__(self, min, max, distribution, seed=None):
        '''
        Constructor.
        
        :param int min: Minimum value (inclusive).
        :param int max: Maximum value (exclusive).
        :param string distribution: One of the following values:
            'uniform': All values between min and max are
            equally likely to be sampled.
            'log10': A logarithmic bias of base 10 is used.
        :param object seed: Seed to the random number generator.
        '''
        super().__init__(seed)
        
        if min > max:
            raise ValueError('min cannot be greater than max (min={}, max={}).'.format(min, max))
        if distribution not in ['uniform', 'log10']:
            raise ValueError('distribution must be uniform or log10, not {}.'.format(distribution))
        if distribution == 'log10':
            if min <= 0:
                raise ValueError('When distribution is logarithmic, min must be a positive number (not {}).'.format(min))
            if max <= 0:
                raise ValueError('When distribution is logarithmic, max must be a positive number (not {}).'.format(max))
        
        self.min = min
        self.max = max
        self.distribution = distribution
        self.rng = random.Random(seed)
        self.method = {
            'uniform': lambda:self.rng.uniform(self.min, self.max),
            'log10':   lambda:10**self.rng.uniform(math.log10(self.min), math.log10(self.max))
            }[distribution]
    
    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.value = self.method()
        self.initialised = True
