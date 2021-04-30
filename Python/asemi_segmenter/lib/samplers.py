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
        self.mutable_samplers = list()
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
        self.mutable_samplers.append(sampler)
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
            'log': A logarithmic bias is used.
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
        self.mutable_samplers.append(sampler)
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
        chosen_sampler = self.rand.choice(self.mutable_samplers)
        chosen_sampler.resample()

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
        self.initialised = True

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
        if distribution not in ['uniform', 'log2']:
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
            'log2':    (lambda:int(math.log2(self.max)) - int(math.log2(self.min)) + 1)
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
        self.initialised = True

    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.set_value({
            'uniform': (lambda:self.rng.randint(self.min, self.max)),
            'log2':    (lambda:2**self.rng.randint(int(math.log2(self.min)), int(math.log2(self.max))))
            }[self.distribution]())


#########################################
class FloatSampler(Sampler):
    '''Sample a random float.'''

    #########################################
    def __init__(self, min, max, divisions, distribution, seed=None):
        '''
        Constructor.

        :param int min: Minimum value (inclusive).
        :param int max: Maximum value (exclusive).
        :param int divisions: Number of samples to take from range.
        :param string distribution: One of the following values:
            'uniform': All values between min and max are
            equally likely to be sampled.
            'log': A logarithmic bias is used.
        :param object seed: Seed to the random number generator.
        '''
        super().__init__(seed)

        if min > max:
            raise ValueError('min cannot be greater than max (min={}, max={}).'.format(min, max))
        if distribution == 'log':
            if min <= 0:
                raise ValueError('When distribution is logarithmic, min must be a positive number (not {}).'.format(min))
            if max <= 0:
                raise ValueError('When distribution is logarithmic, max must be a positive number (not {}).'.format(max))
        if divisions < 1:
            raise ValueError('divisions must be positive.')

        self.min = min
        self.max = max
        self.divisions = divisions
        self.distribution = distribution
        self.rng = random.Random(seed)
        if self.distribution == 'uniform':
            # base = (max - min)/(divs - 1)
            # y = min + base*i
            if self.divisions > 1:
                self.base = (self.max - self.min)/(self.divisions - 1)
            else:
                self.base = 0.0
        elif self.distribution == 'log':
            # base = (log(max) - log(min))/(divs - 1)
            # y = min*exp(base*i)
            if self.divisions > 1:
                self.base = (math.log(self.max) - math.log(self.min))/(self.divisions - 1)
            else:
                self.base = 0.0

    #########################################
    def get_sample_space_size(self):
        '''
        Get the number of different values possible.

        :return: The number of different values.
        :rtype: int
        '''
        return self.divisions

    #########################################
    def set_value(self, value):
        '''
        Set current value.

        :param generic value: The new value.
        '''
        if value < self.min or value > self.max:
            raise ValueError('New value is not within given range.')
        self.value = value
        self.initialised = True

    #########################################
    def resample(self):
        '''
        Generate a new random value.
        '''
        self.set_value({
            'uniform': (lambda:self.min + self.base*self.rng.randint(0, self.divisions - 1)),
            'log':     (lambda:self.min*math.exp(self.base*self.rng.randint(0, self.divisions - 1)))
            }[self.distribution]())
