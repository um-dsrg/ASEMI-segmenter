'''Random generators for the purpose of hyperparameter random search.'''

import random
import math


#########################################
class SamplerFactory(object):
    '''Factory for samplers to collectively resample.'''
    
    #########################################
    def __init__(self, seed):
        '''
        Constructor.
        
        :param object seed: Seed to the random number generator.
        '''
        self.seed = seed
        self.rand = random.Random(seed)
        self.samplers = list()
        self.named_samplers = dict()
    
    #########################################
    def create_constant_sampler(self, value, name=None):
        '''
        Create a ConstantSampler.
        
        :param generic value: Value to return.
        :param str name: Name to use to refer to created sampler.
            If None then will not be referrable.
        :return: The Sampler object.
        :rtype: ConstantSampler.
        '''
        sampler = ConstantSampler(
            value,
            seed=self.rand.random()
            )
        self.samplers.append(sampler)
        if name is not None:
            if name in self.named_samplers:
                raise ValueError('The variable name {} already exists.'.format(name))
            self.named_samplers[name] = sampler
        return sampler
    
    #########################################
    def create_integer_sampler(self, min, max, distribution, name=None):
        '''
        Create an IntegerSampler.
        
        :param int min: Minimum value (inclusive).
        :param int max: Maximum value (inclusive).
        :param string distribution: One of the following values:
            'uniform': All values between min and max are
            equally likely to be sampled.
            'log2': Only powers of 2 between min and max will be sampled.
        :param str name: Name to use to refer to created sampler.
            If None then will not be referrable.
        :return: The Sampler object.
        :rtype: IntegerSampler.
        '''
        sampler = IntegerSampler(
            min,
            max,
            distribution,
            seed=self.rand.random()
            )
        self.samplers.append(sampler)
        if name is not None:
            if name in self.named_samplers:
                raise ValueError('The variable name {} already exists.'.format(name))
            self.named_samplers[name] = sampler
        return sampler
    
    #########################################
    def create_float_sampler(self, min, max, decimal_places, distribution, name=None):
        '''
        Create a FloatSampler.
        
        :param float min: Minimum value (inclusive).
        :param float max: Maximum value (exclusive).
        :param int decimal_places: Number of digits after the decimal point.
        :param string distribution: One of the following values:
            'uniform': All values between min and max are
            equally likely to be sampled.
            'log10': A logarithmic bias of base 10 is used.
        :param str name: Name to use to refer to created sampler.
            If None then will not be referrable.
        :return: The Sampler object.
        :rtype: FloatSampler.
        '''
        sampler = FloatSampler(
            min,
            max,
            decimal_places,
            distribution,
            seed=self.rand.random()
            )
        self.samplers.append(sampler)
        if name is not None:
            if name in self.named_samplers:
                raise ValueError('The variable name {} already exists.'.format(name))
            self.named_samplers[name] = sampler
        return sampler
    
    #########################################
    def get_named_sampler(self, name, expected_type):
        '''
        Get a named sampler that was previously created.
        
        :param str name: The name of the sampler.
        :param str expected_type: The expected type of the sampler. Can
            be 'float' or 'integer'.
        '''
        if name not in self.named_samplers:
            raise ValueError('The variable name {} was not defined.'.format(name))
        if expected_type == 'integer':
            if not isinstance(self.named_samplers[name], IntegerSampler):
                raise ValueError('Parameter {} can only refer to an integer variable.')
        elif expected_type == 'float':
            if not isinstance(self.named_samplers[name], FloatSampler):
                raise ValueError('Parameter {} can only refer to a float variable.')
        return self.named_samplers[name]
    
    #########################################
    def get_sample_space_size(self):
        '''
        Get the number of different values possible with all the samplers combined.
        
        :return: The number of different values.
        :rtype: int
        '''
        size = 1
        for sampler in self.samplers:
            size *= sampler.get_sample_space_size()
        return size
        
    #########################################
    def resample_random_one(self):
        '''
        Resample a randomly selected generated sampler.
        '''
        self.rand.choice(self.samplers).resample()
    
    #########################################
    def resample_all(self):
        '''
        Resample all generated samplers.
        '''
        for sampler in self.samplers:
            sampler.resample()
    

#########################################
class Sampler(object):
    '''Super class for samplers.'''
    
    #########################################
    def __init__(self, seed):
        '''
        Constructor.
        :param object seed: Seed to the random number generator.
        '''
        self.seed = seed
        self.initialised = False
        self.value = None
    
    #########################################
    def get_sample_space_size(self):
        '''
        Get the number of different values possible.
        
        :return: The number of different values.
        :rtype: int
        '''
        raise NotImplementedError()
    
    #########################################
    def set_value(self, value):
        '''
        Set current value.
        
        :param generic value: The new value.
        '''
        raise NotImplementedError()
    
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
    def get_sample_space_size(self):
        '''
        Get the number of different values possible.
        
        :return: The number of different values.
        :rtype: int
        '''
        return 1
    
    #########################################
    def set_value(self, value):
        '''
        Set current value.
        
        :param generic value: The new value.
        '''
        pass
    
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
    
    #########################################
    def get_sample_space_size(self):
        '''
        Get the number of different values possible.
        
        :return: The number of different values.
        :rtype: int
        '''
        return {
            'uniform': (lambda:self.max - self.min + 1),
            'log2':    (lambda:int(math.log2(self.max) - math.log2(self.min) + 1))
            }[self.distribution]()
    
    #########################################
    def set_value(self, value):
        '''
        Set current value.
        
        :param generic value: The new value.
        '''
        if value < self.min or value > self.max:
            raise ValueError('New value is not within given range.')
        if self.distribution == 'log2':
            if math.log2(value)%1 != 0:
                raise ValueError('New value is not a power of 2.')
        self.value = value
    
    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.set_value({
            'uniform': (lambda:self.rng.randint(self.min, self.max)),
            'log2':    (lambda:2**self.rng.randint(int(math.log2(self.min)), int(math.log2(self.max))))
            }[self.distribution]())
        self.initialised = True


#########################################
class FloatSampler(Sampler):
    '''Sample a random float.'''
    
    #########################################
    def __init__(self, min, max, decimal_places, distribution, seed=None):
        '''
        Constructor.
        
        :param int min: Minimum value (inclusive).
        :param int max: Maximum value (exclusive).
        :param int decimal_places: Number of digits after the decimal point.
        :param string distribution: One of the following values:
            'uniform': All values between min and max are
            equally likely to be sampled.
            'log10': A logarithmic bias of base 10 is used.
        :param object seed: Seed to the random number generator.
        '''
        super().__init__(seed)
        
        def get_num_decimal_places(num):
            '''Get the number of decimal places in a number.'''
            fractional = str(float(num)).split('.')[1]
            if fractional == '0':
                return 0
            else:
                return len(fractional)
        
        if min > max:
            raise ValueError('min cannot be greater than max (min={}, max={}).'.format(min, max))
        if get_num_decimal_places(min) > decimal_places:
            raise ValueError('min cannot have more decimal places than decimal_places (min={}, decimal_places={}).'.format(min, decimal_places))
        if get_num_decimal_places(max) > decimal_places:
            raise ValueError('max cannot have more decimal places than decimal_places (max={}, decimal_places={}).'.format(max, decimal_places))
        if distribution not in ['uniform', 'log10']:
            raise ValueError('distribution must be uniform or log10, not {}.'.format(distribution))
        if distribution == 'log10':
            if min <= 0:
                raise ValueError('When distribution is logarithmic, min must be a positive number (not {}).'.format(min))
            if max <= 0:
                raise ValueError('When distribution is logarithmic, max must be a positive number (not {}).'.format(max))
        
        self.min = min
        self.max = max
        self.decimal_places = decimal_places
        self.distribution = distribution
        self.rng = random.Random(seed)
    
    #########################################
    def get_sample_space_size(self):
        '''
        Get the number of different values possible.
        
        :return: The number of different values.
        :rtype: int
        '''
        # The 3 decimal place numbers between 1.0 and 2.0 is:
        # 1.0, 1.001, 1.002, ..., 1.998, 1.999
        # The length of this list is (amount of whole numbers)*(10**dp)
        #  = max(2 - 1, 1)*(10**3) = 1000
        # This is the trivial case when both fractional numbers are 0.
        #
        # The 3 decimal place numbers between 0.01 and 0.1 is:
        # 0.010, 0.011, 0.012, ..., 0.098, 0.099
        # The length of this list is found as follows:
        # Convert each limit's fractional part to an integer with the right number of dp
        #  0.01 -> 01 -> 010 -> 10
        #  0.1  -> 1  -> 100 -> 100
        # Then: (amount of whole numbers)*(amount of fractional numbers)
        #  = max(0 - 0, 1)*(100 - 10) = 90
        
        def get_fractional_part(num):
            '''Get the part of the number after the decimal point as a string.'''
            return str(float(num)).split('.')[1]
        
        min_str_frac = get_fractional_part(self.min)
        min_str_frac += '0'*(self.decimal_places - len(min_str_frac))
        
        max_str_frac = get_fractional_part(self.max)
        max_str_frac += '0'*(self.decimal_places - len(max_str_frac))
        
        return max(int(self.max) - int(self.min), 1)*(int(max_str_frac) - int(min_str_frac))
    
    #########################################
    def set_value(self, value):
        '''
        Set current value.
        
        :param generic value: The new value.
        '''
        if value < self.min or value > self.max:
            raise ValueError('New value is not within given range.')
        self.value = value
    
    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.set_value({
            'uniform': (lambda:round(self.rng.uniform(self.min, self.max), self.decimal_places)),
            'log10':   (lambda:round(10**self.rng.uniform(math.log10(self.min), math.log10(self.max)), self.decimal_places))
            }[self.distribution]())
        self.initialised = True
