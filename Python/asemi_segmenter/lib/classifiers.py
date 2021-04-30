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

'''Module containing classifiers to classify voxels into labels.'''

import os
import random
import warnings
import contextlib
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import sklearn.ensemble
with warnings.catch_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.simplefilter('ignore', category=FutureWarning)
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import samplers
from asemi_segmenter.lib import times


#########################################
def load_classifier_from_config(labels, config, max_batch_memory, use_gpu, sklearn_model=None, sampler_factory=None):
    '''
    Load a classifier from a configuration dictionary.

    :param list labels: List of labels to recognise.
    :param dict config: Configuration of the classifier.
    :param float max_batch_memory: The maximum number of gigabytes of memory to use.
    :param bool use_gpu: Whether to use the GPU.
    :param sklearn_model sklearn_model: Trained sklearn model if available.
    :param samplers.SamplerFactory sampler_factory: The factory to use to create samplers
        for the featuriser parameters. If None then only constant parameters can be used.
    :return: A classifier object.
    :rtype: Classifier
    '''
    validations.validate_json_with_schema_file(config, 'classifier.json')

    if sampler_factory is not None and sklearn_model is not None:
        raise ValueError('Cannot generate hyperparameters from samples when sklearn model is provided.')

    if config['type'] == 'logistic_regression':
        c = None
        max_iter = None

        if sampler_factory is not None:
            if isinstance(config['params']['C'], dict):
                c = sampler_factory.create_float_sampler(
                    config['params']['C']['min'],
                    config['params']['C']['max'],
                    config['params']['C']['divisions'],
                    config['params']['C']['distribution']
                    )
            elif isinstance(config['params']['C'], str):
                c = sampler_factory.get_named_sampler(
                    config['params']['C'],
                    'float'
                    )
            else:
                c = sampler_factory.create_constant_sampler(
                    config['params']['C']
                    )
        else:
            if isinstance(config['params']['C'], dict):
                raise ValueError('C must be a constant not a range.')
            c = config['params']['C']

        if sampler_factory is not None:
            if isinstance(config['params']['max_iter'], dict):
                max_iter = sampler_factory.create_integer_sampler(
                    config['params']['max_iter']['min'],
                    config['params']['max_iter']['max'],
                    config['params']['max_iter']['distribution']
                    )
            elif isinstance(config['params']['max_iter'], str):
                max_iter = sampler_factory.get_named_sampler(
                    config['params']['max_iter'],
                    'integer'
                    )
            else:
                max_iter = sampler_factory.create_constant_sampler(
                    config['params']['max_iter']
                    )
        else:
            if isinstance(config['params']['max_iter'], dict):
                raise ValueError('max_iter must be a constant not a range.')
            max_iter = config['params']['max_iter']

        if sklearn_model is not None:
            if not isinstance(sklearn_model, sklearn.pipeline.Pipeline):
                raise ValueError('sklearn_model is invalid as it is not a pipeline.')
            if set(sklearn_model.named_steps.keys()) != {'preprocessor', 'classifier'}:
                raise ValueError('sklearn_model is invalid as pipeline named steps does not have the expected names.')
            if not isinstance(sklearn_model.named_steps['preprocessor'], sklearn.preprocessing.StandardScaler):
                raise ValueError('sklearn_model is invalid as preprocessor type is not StandardScaler.')
            if not isinstance(sklearn_model.named_steps['classifier'], sklearn.linear_model.LogisticRegression):
                raise ValueError('sklearn_model is invalid as classifier type is not LogisticRegression.')
            if sklearn_model.classes_.size != len(labels):
                raise ValueError(
                    'sklearn_model is invalid as the number of classes is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        len(labels),
                        sklearn_model.classes_.size
                        )
                    )
            if sklearn_model.named_steps['classifier'].C != c:
                raise ValueError('sklearn_model is invalid as C is not as declared.')
            if sklearn_model.named_steps['classifier'].max_iter != max_iter:
                raise ValueError('sklearn_model is invalid as max_iter is not as declared.')

        return LogisticRegressionClassifier(labels, c, max_iter, sklearn_model)

    elif config['type'] == 'neural_network':
        hidden_layer_sizes = list()
        alpha = None
        batch_size = None
        max_iter = None

        if sampler_factory is not None:
            for i in range(len(config['params']['hidden_layer_sizes'])):
                if isinstance(config['params']['hidden_layer_sizes'][i], dict):
                    hidden_layer_sizes.append(sampler_factory.create_integer_sampler(
                        config['params']['hidden_layer_sizes'][i]['min'],
                        config['params']['hidden_layer_sizes'][i]['max'],
                        config['params']['hidden_layer_sizes'][i]['distribution']
                        ))
                elif isinstance(config['params']['hidden_layer_sizes'][i], str):
                    hidden_layer_sizes.append(sampler_factory.get_named_sampler(
                        config['params']['hidden_layer_sizes'][i],
                        'integer'
                        ))
                else:
                    hidden_layer_sizes.append(sampler_factory.create_constant_sampler(
                        config['params']['hidden_layer_sizes'][i]
                        ))
        else:
            for i in range(len(config['params']['hidden_layer_sizes'])):
                if isinstance(config['params']['hidden_layer_sizes'][i], dict):
                    raise ValueError('hidden_layer_sizes item {} must be a constant not a range.'.format(i))
                hidden_layer_sizes.append(config['params']['hidden_layer_sizes'][i])

        if sampler_factory is not None:
            if isinstance(config['params']['alpha'], dict):
                alpha = sampler_factory.create_float_sampler(
                    config['params']['alpha']['min'],
                    config['params']['alpha']['max'],
                    config['params']['alpha']['divisions'],
                    config['params']['alpha']['distribution']
                    )
            elif isinstance(config['params']['alpha'], str):
                alpha = sampler_factory.get_named_sampler(
                    config['params']['alpha'],
                    'float'
                    )
            else:
                alpha = sampler_factory.create_constant_sampler(
                    config['params']['alpha']
                    )
        else:
            if isinstance(config['params']['alpha'], dict):
                raise ValueError('alpha must be a constant not a range.')
            alpha = config['params']['alpha']

        if sampler_factory is not None:
            if isinstance(config['params']['batch_size'], dict):
                batch_size = sampler_factory.create_integer_sampler(
                    config['params']['batch_size']['min'],
                    config['params']['batch_size']['max'],
                    config['params']['batch_size']['distribution']
                    )
            elif isinstance(config['params']['batch_size'], str):
                batch_size = sampler_factory.get_named_sampler(
                    config['params']['batch_size'],
                    'integer'
                    )
            else:
                batch_size = sampler_factory.create_constant_sampler(
                    config['params']['batch_size']
                    )
        else:
            if isinstance(config['params']['batch_size'], dict):
                raise ValueError('batch_size must be a constant not a range.')
            batch_size = config['params']['batch_size']

        if sampler_factory is not None:
            if isinstance(config['params']['max_iter'], dict):
                max_iter = sampler_factory.create_integer_sampler(
                    config['params']['max_iter']['min'],
                    config['params']['max_iter']['max'],
                    config['params']['max_iter']['distribution']
                    )
            elif isinstance(config['params']['max_iter'], str):
                max_iter = sampler_factory.get_named_sampler(
                    config['params']['max_iter'],
                    'integer'
                    )
            else:
                max_iter = sampler_factory.create_constant_sampler(
                    config['params']['max_iter']
                    )
        else:
            if isinstance(config['params']['max_iter'], dict):
                raise ValueError('max_iter must be a constant not a range.')
            max_iter = config['params']['max_iter']

        if sklearn_model is not None:
            if not isinstance(sklearn_model, sklearn.pipeline.Pipeline):
                raise ValueError('sklearn_model is invalid as it is not a pipeline.')
            if set(sklearn_model.named_steps.keys()) != {'preprocessor', 'classifier'}:
                raise ValueError('sklearn_model is invalid as pipeline named steps does not have the expected names.')
            if not isinstance(sklearn_model.named_steps['preprocessor'], sklearn.preprocessing.StandardScaler):
                raise ValueError('sklearn_model is invalid as preprocessor type is not StandardScaler.')
            if not isinstance(sklearn_model.named_steps['classifier'], sklearn.neural_network.MLPClassifier):
                raise ValueError('sklearn_model is invalid as classifier type is not MLPClassifier.')
            if sklearn_model.classes_.size != len(labels):
                raise ValueError(
                    'sklearn_model is invalid as the number of classes is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        len(labels),
                        sklearn_model.classes_.size
                        )
                    )
            if sklearn_model.named_steps['classifier'].hidden_layer_sizes != hidden_layer_sizes:
                raise ValueError('sklearn_model is invalid as hidden_layer_sizes is not as declared.')
            if sklearn_model.named_steps['classifier'].alpha != alpha:
                raise ValueError('sklearn_model is invalid as alpha is not as declared.')
            if sklearn_model.named_steps['classifier'].batch_size != batch_size:
                raise ValueError('sklearn_model is invalid as batch_size is not as declared.')
            if sklearn_model.named_steps['classifier'].max_iter != max_iter:
                raise ValueError('sklearn_model is invalid as max_iter is not as declared.')

        return NeuralNetworkClassifier(labels, hidden_layer_sizes, alpha, batch_size, max_iter, sklearn_model)

    elif config['type'] == 'decision_tree':
        max_depth = None
        min_samples_leaf = None

        if sampler_factory is not None:
            if isinstance(config['params']['max_depth'], dict):
                max_depth = sampler_factory.create_integer_sampler(
                    config['params']['max_depth']['min'],
                    config['params']['max_depth']['max'],
                    config['params']['max_depth']['distribution']
                    )
            elif isinstance(config['params']['max_depth'], str):
                max_depth = sampler_factory.get_named_sampler(
                    config['params']['max_depth'],
                    'integer'
                    )
            else:
                max_depth = sampler_factory.create_constant_sampler(
                    config['params']['max_depth']
                    )
        else:
            if isinstance(config['params']['max_depth'], dict):
                raise ValueError('max_depth must be a constant not a range.')
            max_depth = config['params']['max_depth']

        if sampler_factory is not None:
            if isinstance(config['params']['min_samples_leaf'], dict):
                min_samples_leaf = sampler_factory.create_integer_sampler(
                    config['params']['min_samples_leaf']['min'],
                    config['params']['min_samples_leaf']['max'],
                    config['params']['min_samples_leaf']['distribution']
                    )
            elif isinstance(config['params']['min_samples_leaf'], str):
                min_samples_leaf = sampler_factory.get_named_sampler(
                    config['params']['min_samples_leaf'],
                    'integer'
                    )
            else:
                min_samples_leaf = sampler_factory.create_constant_sampler(
                    config['params']['min_samples_leaf']
                    )
        else:
            if isinstance(config['params']['min_samples_leaf'], dict):
                raise ValueError('min_samples_leaf must be a constant not a range.')
            min_samples_leaf = config['params']['min_samples_leaf']

        if sampler_factory is not None:
            if not isinstance(sklearn_model, sklearn.pipeline.Pipeline):
                raise ValueError('sklearn_model is invalid as it is not a pipeline.')
            if set(sklearn_model.named_steps.keys()) != {'preprocessor', 'classifier'}:
                raise ValueError('sklearn_model is invalid as pipeline named steps does not have the expected names.')
            if not isinstance(sklearn_model.named_steps['preprocessor'], sklearn.preprocessing.StandardScaler):
                raise ValueError('sklearn_model is invalid as preprocessor type is not StandardScaler.')
            if not isinstance(sklearn_model.named_steps['classifier'], sklearn.tree.DecisionTreeClassifier):
                raise ValueError('sklearn_model is invalid as classifier type is not DecisionTreeClassifier.')
            if sklearn_model.classes_.size != len(labels):
                raise ValueError(
                    'sklearn_model is invalid as the number of classes is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        len(labels),
                        sklearn_model.classes_.size
                        )
                    )
            if sklearn_model.named_steps['classifier'].max_depth != max_depth:
                raise ValueError('sklearn_model is invalid as max_depth is not as declared.')
            if sklearn_model.named_steps['classifier'].min_samples_leaf != min_samples_leaf:
                raise ValueError('sklearn_model is invalid as min_samples_leaf is not as declared.')

        return DecisionTreeClassifier(labels, max_depth, min_samples_leaf, sklearn_model)

    elif config['type'] == 'random_forest':
        n_estimators = None
        max_depth = None
        min_samples_leaf = None

        if sampler_factory is not None:
            if isinstance(config['params']['n_estimators'], dict):
                n_estimators = sampler_factory.create_integer_sampler(
                    config['params']['n_estimators']['min'],
                    config['params']['n_estimators']['max'],
                    config['params']['n_estimators']['distribution']
                    )
            elif isinstance(config['params']['n_estimators'], str):
                n_estimators = sampler_factory.get_named_sampler(
                    config['params']['n_estimators'],
                    'integer'
                    )
            else:
                n_estimators = sampler_factory.create_constant_sampler(
                    config['params']['n_estimators']
                    )
        else:
            if isinstance(config['params']['n_estimators'], dict):
                raise ValueError('n_estimators must be a constant not a range.')
            n_estimators = config['params']['n_estimators']

        if sampler_factory is not None:
            if isinstance(config['params']['max_depth'], dict):
                max_depth = sampler_factory.create_integer_sampler(
                    config['params']['max_depth']['min'],
                    config['params']['max_depth']['max'],
                    config['params']['max_depth']['distribution']
                    )
            elif isinstance(config['params']['max_depth'], str):
                max_depth = sampler_factory.get_named_sampler(
                    config['params']['max_depth'],
                    'integer'
                    )
            else:
                max_depth = sampler_factory.create_constant_sampler(
                    config['params']['max_depth']
                    )
        else:
            if isinstance(config['params']['max_depth'], dict):
                raise ValueError('max_depth must be a constant not a range.')
            max_depth = config['params']['max_depth']

        if sampler_factory is not None:
            if isinstance(config['params']['min_samples_leaf'], dict):
                min_samples_leaf = sampler_factory.create_integer_sampler(
                    config['params']['min_samples_leaf']['min'],
                    config['params']['min_samples_leaf']['max'],
                    config['params']['min_samples_leaf']['distribution']
                    )
            elif isinstance(config['params']['min_samples_leaf'], str):
                min_samples_leaf = sampler_factory.get_named_sampler(
                    config['params']['min_samples_leaf'],
                    'integer'
                    )
            else:
                min_samples_leaf = sampler_factory.create_constant_sampler(
                    config['params']['min_samples_leaf']
                    )
        else:
            if isinstance(config['params']['min_samples_leaf'], dict):
                raise ValueError('min_samples_leaf must be a constant not a range.')
            min_samples_leaf = config['params']['min_samples_leaf']

        if sklearn_model is not None:
            if not isinstance(sklearn_model, sklearn.pipeline.Pipeline):
                raise ValueError('sklearn_model is invalid as it is not a pipeline.')
            if set(sklearn_model.named_steps.keys()) != {'preprocessor', 'classifier'}:
                raise ValueError('sklearn_model is invalid as pipeline named steps does not have the expected names.')
            if not isinstance(sklearn_model.named_steps['preprocessor'], sklearn.preprocessing.StandardScaler):
                raise ValueError('sklearn_model is invalid as preprocessor type is not StandardScaler.')
            if not isinstance(sklearn_model.named_steps['classifier'], sklearn.ensemble.RandomForestClassifier):
                raise ValueError('sklearn_model is invalid as classifier type is not RandomForestClassifier.')
            if sklearn_model.classes_.size != len(labels):
                raise ValueError(
                    'sklearn_model is invalid as the number of classes is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        len(labels),
                        sklearn_model.classes_.size
                        )
                    )
            if sklearn_model.named_steps['classifier'].n_estimators != n_estimators:
                raise ValueError('sklearn_model is invalid as n_estimators is not as declared.')
            if sklearn_model.named_steps['classifier'].max_depth != max_depth:
                raise ValueError('sklearn_model is invalid as max_depth is not as declared.')
            if sklearn_model.named_steps['classifier'].min_samples_leaf != min_samples_leaf:
                raise ValueError('sklearn_model is invalid as min_samples_leaf is not as declared.')

        return RandomForestClassifier(labels, n_estimators, max_depth, min_samples_leaf, sklearn_model)

    elif config['type'] == 'tensorflow_neural_network':
        hidden_layer_sizes = list()
        dropout_rate = None
        init_stddev = None
        batch_size = None
        max_iter = None

        if sampler_factory is not None:
            for i in range(len(config['params']['hidden_layer_sizes'])):
                if isinstance(config['params']['hidden_layer_sizes'][i], dict):
                    hidden_layer_sizes.append(sampler_factory.create_integer_sampler(
                        config['params']['hidden_layer_sizes'][i]['min'],
                        config['params']['hidden_layer_sizes'][i]['max'],
                        config['params']['hidden_layer_sizes'][i]['distribution']
                        ))
                elif isinstance(config['params']['hidden_layer_sizes'][i], str):
                    hidden_layer_sizes.append(sampler_factory.get_named_sampler(
                        config['params']['hidden_layer_sizes'][i],
                        'integer'
                        ))
                else:
                    hidden_layer_sizes.append(sampler_factory.create_constant_sampler(
                        config['params']['hidden_layer_sizes'][i]
                        ))
        else:
            for i in range(len(config['params']['hidden_layer_sizes'])):
                if isinstance(config['params']['hidden_layer_sizes'][i], dict):
                    raise ValueError('hidden_layer_sizes item {} must be a constant not a range.'.format(i))
                hidden_layer_sizes.append(config['params']['hidden_layer_sizes'][i])

        if sampler_factory is not None:
            if isinstance(config['params']['dropout_rate'], dict):
                dropout_rate = sampler_factory.create_float_sampler(
                    config['params']['dropout_rate']['min'],
                    config['params']['dropout_rate']['max'],
                    config['params']['dropout_rate']['divisions'],
                    config['params']['dropout_rate']['distribution']
                    )
            elif isinstance(config['params']['dropout_rate'], str):
                dropout_rate = sampler_factory.get_named_sampler(
                    config['params']['dropout_rate'],
                    'float'
                    )
            else:
                dropout_rate = sampler_factory.create_constant_sampler(
                    config['params']['dropout_rate']
                    )
        else:
            if isinstance(config['params']['dropout_rate'], dict):
                raise ValueError('dropout_rate must be a constant not a range.')
            dropout_rate = config['params']['dropout_rate']

        if sampler_factory is not None:
            if isinstance(config['params']['init_stddev'], dict):
                init_stddev = sampler_factory.create_float_sampler(
                    config['params']['init_stddev']['min'],
                    config['params']['init_stddev']['max'],
                    config['params']['init_stddev']['divisions'],
                    config['params']['init_stddev']['distribution']
                    )
            elif isinstance(config['params']['init_stddev'], str):
                init_stddev = sampler_factory.get_named_sampler(
                    config['params']['init_stddev'],
                    'float'
                    )
            else:
                init_stddev = sampler_factory.create_constant_sampler(
                    config['params']['init_stddev']
                    )
        else:
            if isinstance(config['params']['init_stddev'], dict):
                raise ValueError('init_stddev must be a constant not a range.')
            init_stddev = config['params']['init_stddev']

        if sampler_factory is not None:
            if isinstance(config['params']['batch_size'], dict):
                batch_size = sampler_factory.create_integer_sampler(
                    config['params']['batch_size']['min'],
                    config['params']['batch_size']['max'],
                    config['params']['batch_size']['distribution']
                    )
            elif isinstance(config['params']['batch_size'], str):
                batch_size = sampler_factory.get_named_sampler(
                    config['params']['batch_size'],
                    'integer'
                    )
            else:
                batch_size = sampler_factory.create_constant_sampler(
                    config['params']['batch_size']
                    )
        else:
            if isinstance(config['params']['batch_size'], dict):
                raise ValueError('batch_size must be a constant not a range.')
            batch_size = config['params']['batch_size']

        if sampler_factory is not None:
            if isinstance(config['params']['max_iter'], dict):
                max_iter = sampler_factory.create_integer_sampler(
                    config['params']['max_iter']['min'],
                    config['params']['max_iter']['max'],
                    config['params']['max_iter']['distribution']
                    )
            elif isinstance(config['params']['max_iter'], str):
                max_iter = sampler_factory.get_named_sampler(
                    config['params']['max_iter'],
                    'integer'
                    )
            else:
                max_iter = sampler_factory.create_constant_sampler(
                    config['params']['max_iter']
                    )
        else:
            if isinstance(config['params']['max_iter'], dict):
                raise ValueError('max_iter must be a constant not a range.')
            max_iter = config['params']['max_iter']

        if sklearn_model is not None:
            if not isinstance(sklearn_model, dict):
                raise ValueError('sklearn_model is invalid as it is not a dictionary.')
            if set(sklearn_model.keys()) != {'preprocessor', 'classifier'}:
                raise ValueError('sklearn_model is invalid as dictionary keys are not as expected.')
            if not isinstance(sklearn_model['preprocessor'], sklearn.preprocessing.StandardScaler):
                raise ValueError('sklearn_model is invalid as preprocessor type is not StandardScaler.')
            if not isinstance(sklearn_model['classifier'], dict):
                raise ValueError('sklearn_model is invalid as classifier data is not a dictionary.')
            if sklearn_model['classifier']['num_outputs'] != len(labels):
                raise ValueError(
                    'sklearn_model is invalid as the number of classes is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        len(labels),
                        sklearn_model['classifier']['num_outputs']
                        )
                    )
            if sklearn_model['classifier']['hidden_layer_sizes'] != hidden_layer_sizes:
                raise ValueError('sklearn_model is invalid as hidden_layer_sizes is not as declared.')
            if sklearn_model['classifier']['dropout_rate'] != dropout_rate:
                raise ValueError('sklearn_model is invalid as dropout_rate is not as declared.')
            if sklearn_model['classifier']['init_stddev'] != init_stddev:
                raise ValueError('sklearn_model is invalid as init_stddev is not as declared.')
            if sklearn_model['classifier']['batch_size'] != batch_size:
                raise ValueError('sklearn_model is invalid as batch_size is not as declared.')
            if sklearn_model['classifier']['max_iter'] != max_iter:
                raise ValueError('sklearn_model is invalid as max_iter is not as declared.')

        return TensorflowNeuralNetworkClassifier(labels, hidden_layer_sizes, dropout_rate, init_stddev, batch_size, max_iter, max_batch_memory, use_gpu, sklearn_model)

    else:
        raise NotImplementedError('Classifier {} not implemented.'.format(config['type']))


#########################################
class SklearnLikeTensorflowNeuralNet(object):
    '''
    A feedforward neural network implemented in Tensorflow v1.

    This implements the same interface as sklearn's MLPClassifier. It uses
    leaky ReLU as activation functions, Adam as an optimiser, early stopping
    on the validation accuracy with a patience of 3 as a terminating condition,
    and the output is a softmax.
    '''

    #########################################
    @staticmethod
    def load_from_pickle(pickled_obj, max_batch_memory, use_gpu):
        '''
        Construct and load a usable neural network from its pickled version.

        :param dict pickled_obj: The object returned by get_picklable().
        :param float max_batch_memory: The maximum number of gigabytes of memory to use.
        :param bool use_gpu: Whether to use the GPU.
        :rtype: SklearnLikeTensorflowNeuralNet
        :return: A neural network with its model parameters and hyperparameters set.
        '''
        model = SklearnLikeTensorflowNeuralNet(
            hidden_layer_sizes=pickled_obj['hidden_layer_sizes'],
            dropout_rate=pickled_obj['dropout_rate'],
            init_stddev=pickled_obj['init_stddev'],
            validation_fraction=pickled_obj['validation_fraction'],
            batch_size=pickled_obj['batch_size'],
            max_iter=pickled_obj['max_iter'],
            patience=pickled_obj['patience'],
            verbose=pickled_obj['verbose'],
            random_state=pickled_obj['random_state'],
            max_batch_memory=max_batch_memory,
            use_gpu=use_gpu
            )
        model._create_model(pickled_obj['num_inputs'], pickled_obj['num_outputs'])
        model.set_model_params(pickled_obj['params'])
        model._train_history = pickled_obj['train_history']
        return model

    #########################################
    def __init__(self, hidden_layer_sizes, dropout_rate, init_stddev, validation_fraction, batch_size, max_iter, patience, verbose, random_state, max_batch_memory, use_gpu):
        '''
        Constructor.

        :param list hidden_layer_sizes: The ith element represents the number
            of neurons in the ith hidden layer.
        :param float dropout_rate: The fraction of neurons to randomly drop in
            every layer for every training item.
        :param float init_stddev: The standard deviation to use in a normal
            random number generator to set the weights of the neural network.
        :param float validation_fraction: The proportion of training data to set
            aside as validation set for early stopping.
        :param int batch_size: Size of minibatches during optimisation.
        :param int max_iter: Maximum number of iterations. The optimiser
            iterates until the early stopping condition or this number of
            iterations. Note that this determines the number of epochs (how many
            times each data point will be used), not the number of gradient
            steps.
        :param int patience: Early stopping patience.
        :param bool verbose: Whether to print progress messages to stdout.
        :param int random_state: Determines random number generation for weights
            initialization, train-validation split, and batch sampling. Pass an
            int for reproducible results across multiple function calls.
        :param float max_batch_memory: The maximum number of gigabytes of memory to use.
        :param bool use_gpu: Whether to use the GPU.
        '''
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.init_stddev = init_stddev
        self.validation_fraction = validation_fraction
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.max_batch_memory = max_batch_memory
        self.use_gpu = use_gpu
        self._num_inputs = None
        self._num_outputs = None
        self._in_vecs = None
        self._dropout = None
        self._targets = None
        self._params = []
        self._out_probs = None
        self._error = None
        self._optimiser_step = None
        self._init = None
        self._sess = None
        self._train_history = list()
        self._max_batch_size = None

    #########################################
    def _create_model(self, num_inputs, num_outputs):
        '''Internal convenience function for creating the Tensorflow graph.'''
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        graph = tf.Graph()
        with graph.as_default():
            with tf.device('/cpu:0') if not self.use_gpu else contextlib.suppress():
                self._in_vecs = tf.placeholder(tf.float32, [None, num_inputs], 'in_vecs')
                self._targets = tf.placeholder(tf.int32, [None], 'targets')
                self._dropout = tf.placeholder(tf.bool, [], 'dropout')

                self._params = []
                dropout_keep_prob = tf.cond(self._dropout, lambda:tf.constant(1.0-self.dropout_rate, tf.float32), lambda:tf.constant(1.0, tf.float32))

                prev_layer_size = num_inputs
                prev_layer = self._in_vecs
                for (i, layer_size) in enumerate(self.hidden_layer_sizes):
                    with tf.variable_scope('hidden{}'.format(i)):
                        W = tf.get_variable('W', [prev_layer_size, layer_size], tf.float32, tf.zeros_initializer())
                        W_in = tf.placeholder(tf.float32, [prev_layer_size, layer_size], 'W_in')
                        W_setter = tf.assign(W, W_in)
                        self._params.append((W, W_in, W_setter))

                        b = tf.get_variable('b', [layer_size], tf.float32, tf.zeros_initializer())
                        b_in = tf.placeholder(tf.float32, [layer_size], 'b_in')
                        b_setter = tf.assign(b, b_in)
                        self._params.append((b, b_in, b_setter))

                        prev_layer_size = layer_size
                        prev_layer = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(prev_layer, W) + b), dropout_keep_prob)

                with tf.variable_scope('output'):
                    W = tf.get_variable('W', [prev_layer_size, num_outputs], tf.float32, tf.zeros_initializer())
                    W_in = tf.placeholder(tf.float32, [prev_layer_size, num_outputs], 'W_in')
                    W_setter = tf.assign(W, W_in)
                    self._params.append((W, W_in, W_setter))

                    b = tf.get_variable('b', [num_outputs], tf.float32, tf.zeros_initializer())
                    b_in = tf.placeholder(tf.float32, [num_outputs], 'b_in')
                    b_setter = tf.assign(b, b_in)
                    self._params.append((b, b_in, b_setter))

                    logits = tf.matmul(prev_layer, W) + b
                    self._out_probs = tf.nn.softmax(logits)

                self._error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._targets, logits=logits))

                self._optimiser_step = tf.train.AdamOptimizer().minimize(self._error)

                self._init = tf.global_variables_initializer()

                graph.finalize()

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                self._sess = tf.Session(config=config)

        self._max_batch_size = int(self.max_batch_memory*1024**3)//(max(num_inputs, *self.hidden_layer_sizes, num_outputs)*np.dtype(np.float32).itemsize)

    #########################################
    def close(self):
        '''Close the Tensorflow session.'''
        if self._sess is not None:
            self._sess.close()

    #########################################
    def get_model_params(self):
        '''
        Get the weights and biases of the trained neural network in the order of
        layer 1 (closest to input) weights, layer 1 biases, layer 2 weights,
        layer 2 biases, etc.

        :rtype: list
        :return: A list of numpy arrays.
        '''
        return [self._sess.run(p) for (p, p_in, p_setter) in self._params]

    #########################################
    def set_model_params(self, params):
        '''
        Set the model's weights and biases using a list returned by
        get_model_params().

        :param list params: A list of numpy arrays.
        '''
        if self._sess is None:
            num_inputs = params[0].shape[0]
            num_outputs = params[-1].shape[0]
            self._create_model(num_inputs, num_outputs)
        for ((p, p_in, p_setter), p_val) in zip(self._params, params):
            self._sess.run(p_setter, { p_in: p_val })

    #########################################
    def fit(self, X, y):
        '''
        Fit the model to data matrix X and target(s) y.

        :param numpy.ndarray X: The input data (n_samples, n_features).
        :param numpy.ndarray y: The target value class labels (n_samples,).
        '''
        num_inputs = X.shape[1]
        num_outputs = y.max() + 1
        self._create_model(num_inputs, num_outputs)

        self._sess.run(self._init, { })

        rng = random.Random(self.random_state)

        (X_train, X_val, y_train, y_val) = sklearn.model_selection.train_test_split(
            X, y,
            test_size=self.validation_fraction,
            stratify=y,
            random_state=rng.randrange(2**32)
            )

        params_rng = np.random.RandomState(rng.randrange(2**32))
        for (p, p_in, p_setter) in self._params:
            if len(p.get_shape()) == 2:
                self._sess.run(p_setter, {
                    p_in: params_rng.normal(0.0, self.init_stddev, size=p.get_shape().as_list())
                    })

        sgd_rng = np.random.RandomState(rng.randrange(2**32))
        best_val_acc = 0.0
        best_params = None
        epochs_since_last_best_val_acc = 0
        self._train_history = list()
        if self.verbose:
            print('epoch | val. acc. | new best? |    duration')
            print('------+-----------+-----------+------------')
        for epoch in range(1, self.max_iter + 1):
            with times.Timer() as timer:
                indexes = np.arange(len(X_train))
                sgd_rng.shuffle(indexes)
                for i in range(int(np.ceil(len(indexes)/self.batch_size))):
                    minibatch_indexes = indexes[i*self.batch_size:(i+1)*self.batch_size].tolist()
                    self._sess.run([ self._optimiser_step ], {
                        self._in_vecs: X_train[minibatch_indexes],
                        self._dropout: True,
                        self._targets: y_train[minibatch_indexes]
                        })

                predictions = self.predict(X_val)
                val_acc = np.sum(predictions == y_val)/len(X_val)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = self.get_model_params()
                    epochs_since_last_best_val_acc = 0
                else:
                    epochs_since_last_best_val_acc += 1

                self._train_history.append(val_acc)

            if self.verbose:
                print('{: >5d} | {: >9.2%} | {: >9s} | {: >11s}'.format(epoch, val_acc, 'yes' if epochs_since_last_best_val_acc == 0 else 'no', times.get_readable_duration(timer.duration)))

            if epochs_since_last_best_val_acc >= self.patience:
                break

        self.set_model_params(best_params)

    #########################################
    def predict_proba(self, X):
        '''
        Probability estimates.

        :param numpy.ndarray X: The input data.
        :rtype numpy.ndarray
        :return: The predicted probability of the sample for each class in the
            model.
        '''
        predictions = np.empty((len(X), self._num_outputs), np.float32)
        for i in range(int(np.ceil(len(X)/self._max_batch_size))):
            predictions[i*self._max_batch_size:(i+1)*self._max_batch_size] = self._sess.run(
                self._out_probs,
                {
                    self._in_vecs: X[i*self._max_batch_size:(i+1)*self._max_batch_size],
                    self._dropout: False
                    }
                )
        return predictions

    #########################################
    def predict_log_proba(self, X):
        '''
        Return the log of probability estimates.

        :param numpy.ndarray X: The input data.
        :rtype numpy.ndarray
        :return: The predicted log-probability of the sample for each class in
            the model.
        '''
        return np.log(self.predict_proba(X))

    #########################################
    def predict(self, X):
        '''
        Predict using the multi-layer perceptron classifier.

        :param numpy.ndarray X: The input data.
        :rtype numpy.ndarray
        :return: The predicted classes (n_samples,).
        '''
        return np.argmax(self.predict_proba(X), axis=1)

    #########################################
    def get_training_history(self):
        '''
        Get the training curve of the model in terms of validation accuracy for
        each epoch.

        :rtype: list
        :return: The training history.
        '''
        return self._train_history

    #########################################
    def get_picklable(self):
        '''
        Get the classifier's hyperparameters and trained parameters in a
        picklable form.

        :rtype: dict
        :return: The picklable object.
        '''
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'dropout_rate': self.dropout_rate,
            'init_stddev': self.init_stddev,
            'validation_fraction': self.validation_fraction,
            'batch_size': self.batch_size,
            'max_iter': self.max_iter,
            'patience': self.patience,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'num_inputs': self._num_inputs,
            'num_outputs': self._num_outputs,
            'params': self.get_model_params(),
            'train_history': self._train_history
            }


#########################################
class Classifier(object):
    '''Super class for classifiers.'''

    #########################################
    def __init__(self, labels, sklearn_model=None):
        '''
        Constructor.

        :param list labels: List of labels to be classified.
        :param sklearn_model sklearn_model: The pretrained sklearn model to use, if any.
            If None then an untrained model will be created. Otherwise
            it is validated against the given parameters.
        '''
        self.labels = labels
        self.sklearn_model = sklearn_model

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values and resulting sklearn model from the samplers provided.
        '''
        raise NotImplementedError()

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the classifier parameters.
        '''
        raise NotImplementedError()

    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the classifier's parameters.

        :return: The dictionary configuration.
        :rtype: dict
        '''
        raise NotImplementedError()

    #########################################
    def get_params(self):
        '''
        Get the classifier's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        raise NotImplementedError()

    #########################################
    def get_picklable(self):
        '''
        Get the classifier's inner model in a picklable form.

        :return: The picklable object.
        :rtype: object
        '''
        raise NotImplementedError()

    #########################################
    def train(self, training_set, max_processes=1, verbose_training=True):
        '''
        Turn a slice from a volume into a matrix of feature vectors.

        :param int max_processes: The number of concurrent processes to use.
        :param bool verbose_training: Whether to include sklearn's verbose texts.
        :return: A reference to output.
        :rtype: numpy.ndarray
        '''
        self.sklearn_model.named_steps['classifier'].n_jobs = max_processes

        if verbose_training and hasattr(self.sklearn_model.named_steps['classifier'], 'verbose'):
            if isinstance(self.sklearn_model.named_steps['classifier'].verbose, int):
                self.sklearn_model.named_steps['classifier'].verbose = 2
            elif isinstance(self.sklearn_model.named_steps['classifier'].verbose, bool):
                self.sklearn_model.named_steps['classifier'].verbose = True

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)

            self.sklearn_model.fit(
                training_set.get_features_array(),
                training_set.get_labels_array()
                )

        if verbose_training and hasattr(self.sklearn_model.named_steps['classifier'], 'verbose'):
            if isinstance(self.sklearn_model.named_steps['classifier'].verbose, int):
                self.sklearn_model.named_steps['classifier'].verbose = 0
            elif isinstance(self.sklearn_model.named_steps['classifier'].verbose, bool):
                self.sklearn_model.named_steps['classifier'].verbose = False

    #########################################
    def predict_label_probs(self, features_array, max_processes=1):
        '''
        Predict the probability of each label for each feature vector.

        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int max_processes: The number of concurrent processes to use.
        :return: 2D numpy array with each row being the probabilities for the corresponding
            feature vector and each column being a label.
        :rtype: numpy.ndarray
        '''
        self.sklearn_model.named_steps['classifier'].n_jobs = max_processes
        return self.sklearn_model.predict_proba(features_array)

    #########################################
    def predict_label_onehots(self, features_array, max_processes=1):
        '''
        Predict one hot vectors of each label for each feature vector.

        The label with the highest probability will get a 1 in the vectors' corresponding index
        and the other vector elements will be 0.

        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int max_processes: The number of concurrent processes to use.
        :return: 2D numpy array with each row being the one hot vectors for the corresponding
            feature vector and each column being a label.
        :rtype: numpy.ndarray
        '''
        probs = self.predict_label_probs(features_array, max_processes)
        label_indexes = np.argmax(probs, axis=1)
        probs[:, :] = 0.0
        probs[np.arange(probs.shape[0]), label_indexes] = 1.0
        return probs

    #########################################
    def predict_label_indexes(self, features_array, max_processes=1):
        '''
        Predict a label index for each feature vector.

        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int max_processes: The number of concurrent processes to use.
        :return: An array of integers with each item being the label index for the corresponding feature vector.
        :rtype: numpy.ndarray
        '''
        probs = self.predict_label_probs(features_array, max_processes)
        return np.argmax(probs, axis=1)

    #########################################
    def predict_label_names(self, features_array, max_processes=1):
        '''
        Predict a label name for each feature vector.

        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int max_processes: The number of concurrent processes to use.
        :return: A list of string with each item being the label name for the corresponding feature vector.
        :rtype: list
        '''
        label_indexes = self.predict_label_indexes(features_array, max_processes)
        return [self.labels[label_index] for label_index in label_indexes.tolist()]


#########################################
class LogisticRegressionClassifier(Classifier):
    '''
    For sklearn's sklearn.linear_model.LogisticRegression.

    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    '''

    #########################################
    @staticmethod
    def __MAKE_MODEL(c, max_iter):
        '''Make an sklearn model from parameters.'''
        return sklearn.pipeline.Pipeline([
            (
                'preprocessor',
                sklearn.preprocessing.StandardScaler()
                ),
            (
                'classifier',
                sklearn.linear_model.LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    solver='saga',
                    multi_class='multinomial',
                    penalty='l1',
                    class_weight='balanced',
                    verbose=0,
                    random_state=0
                    )
                )
            ])

    #########################################
    def __init__(self, labels, c, max_iter, sklearn_model=None):
        '''
        Constructor.

        :param list labels: List of labels to be classified.
        :param c: The amount to regularise the model such that
            smaller numbers lead to stronger regularisation.
        :type c: float or samplers.Sampler
        :param max_iter: The number of iterations to spend on
            training.
        :type max_iter: int or samplers.Sampler
        :param sklearn_model: The pretrained sklearn model to use, if any.
            If None then an untrained model will be created. Otherwise
            it is validated against the given parameters.
        :type sklearn_model: None or sklearn_LogisticRegression
        '''
        super().__init__(
            labels,
            (
                sklearn_model
                if sklearn_model is not None
                else self.__MAKE_MODEL(c, max_iter)
                if (
                    not isinstance(c, samplers.Sampler)
                    and not isinstance(max_iter, samplers.Sampler)
                    )
                else None
                )
            )

        self.c = None
        self.max_iter = None
        self.c_sampler = None
        self.max_iter_sampler = None
        if isinstance(c, samplers.Sampler):
            self.c_sampler = c
        else:
            self.c = c
        if isinstance(max_iter, samplers.Sampler):
            self.max_iter_sampler = max_iter
        else:
            self.max_iter = max_iter

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values and resulting sklearn model from the samplers provided.
        '''
        self.c = self.c_sampler.get_value()
        self.max_iter = self.max_iter_sampler.get_value()

        self.sklearn_model = self.__MAKE_MODEL(self.c, self.max_iter)

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the classifier parameters.
        '''
        self.c_sampler.set_value(config['params']['C'])
        self.max_iter_sampler.set_value(config['params']['max_iter'])

    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the classifier's parameters.

        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {
            'type': 'logistic_regression',
            'params': {
                'C': self.c,
                'max_iter': self.max_iter
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the classifier's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (self.c, self.max_iter)

    #########################################
    def get_picklable(self):
        '''
        Get the classifier's inner model in a picklable form.

        :return: The picklable object.
        :rtype: object
        '''
        return self.sklearn_model


#########################################
class NeuralNetworkClassifier(Classifier):
    '''
    For sklearn's sklearn.neural_network.MLPClassifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    '''

    #########################################
    @staticmethod
    def __MAKE_MODEL(hidden_layer_sizes, alpha, batch_size, max_iter):
        '''Make an sklearn model from parameters.'''
        return sklearn.pipeline.Pipeline([
            (
                'preprocessor',
                sklearn.preprocessing.StandardScaler()
                ),
            (
                'classifier',
                sklearn.neural_network.MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    activation='relu',
                    solver='adam',
                    batch_size=batch_size,
                    early_stopping=True,
                    validation_fraction=0.1,
                    tol=1e-4,
                    max_iter=max_iter,
                    verbose=False,
                    random_state=0
                    )
                )
            ])

    #########################################
    def __init__(self, labels, hidden_layer_sizes, alpha, batch_size, max_iter, sklearn_model=None):
        '''
        Constructor.

        :param list labels: List of labels to be classified.
        :param hidden_layer_sizes: The amount of neural units in each
            hidden layer.
        :type hidden_layer_sizes: list or samplers.Sampler
        :param alpha: The amount to regularise the model such that
            larger numbers lead to stronger regularisation.
        :type alpha: float or samplers.Sampler
        :param batch_size: The number of training items in each
            minibatch.
        :type batch_size: int or samplers.Sampler
        :param max_iter: The number of iterations to spend on
            training.
        :type max_iter: int or samplers.Sampler
        :param sklearn_model: The pretrained sklearn model to use, if any.
            If None then an untrained model will be created. Otherwise
            it is validated against the given parameters.
        :type sklearn_model: None or sklearn_LogisticRegression
        '''
        super().__init__(
            labels,
            (
                sklearn_model
                if sklearn_model is not None
                else self.__MAKE_MODEL(hidden_layer_sizes, alpha, batch_size, max_iter)
                if (
                    not any(isinstance(hidden_layer_size, samplers.Sampler) for hidden_layer_size in hidden_layer_sizes)
                    and not isinstance(alpha, samplers.Sampler)
                    and not isinstance(batch_size, samplers.Sampler)
                    and not isinstance(max_iter, samplers.Sampler)
                    )
                else None
                )
            )

        self.hidden_layer_sizes = [None]*len(hidden_layer_sizes)
        self.alpha = None
        self.batch_size = None
        self.max_iter = None
        self.hidden_layer_sizes_samplers = [None]*len(hidden_layer_sizes)
        self.alpha_sampler = None
        self.batch_size_sampler = None
        self.max_iter_sampler = None
        for i in range(len(hidden_layer_sizes)):
            if isinstance(hidden_layer_sizes[i], samplers.Sampler):
                self.hidden_layer_sizes_samplers[i] = hidden_layer_sizes[i]
            else:
                self.hidden_layer_sizes[i] = hidden_layer_sizes[i]
        if isinstance(alpha, samplers.Sampler):
            self.alpha_sampler = alpha
        else:
            self.alpha = alpha
        if isinstance(batch_size, samplers.Sampler):
            self.batch_size_sampler = batch_size
        else:
            self.batch_size = batch_size
        if isinstance(max_iter, samplers.Sampler):
            self.max_iter_sampler = max_iter
        else:
            self.max_iter = max_iter

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values and resulting sklearn model from the samplers provided.
        '''
        for i in range(len(self.hidden_layer_sizes)):
            self.hidden_layer_sizes[i] = self.hidden_layer_sizes_samplers[i].get_value()
        self.alpha = self.alpha_sampler.get_value()
        self.batch_size = self.batch_size_sampler.get_value()
        self.max_iter = self.max_iter_sampler.get_value()

        self.sklearn_model = self.__MAKE_MODEL(self.hidden_layer_sizes, self.alpha, self.batch_size, self.max_iter)

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the classifier parameters.
        '''
        for i in range(len(self.hidden_layer_sizes)):
            self.hidden_layer_sizes_samplers[i].set_value(config['params']['hidden_layer_sizes'][i])
        self.alpha_sampler.set_value(config['params']['alpha'])
        self.batch_size_sampler.set_value(config['params']['batch_size'])
        self.max_iter_sampler.set_value(config['params']['max_iter'])

    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the classifier's parameters.

        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {
            'type': 'neural_network',
            'params': {
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'alpha': self.alpha,
                'batch_size': self.batch_size,
                'max_iter': self.max_iter
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the classifier's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (tuple(self.hidden_layer_sizes), self.alpha, self.batch_size, self.max_iter)

    #########################################
    def get_picklable(self):
        '''
        Get the classifier's inner model in a picklable form.

        :return: The picklable object.
        :rtype: object
        '''
        return self.sklearn_model


#########################################
class DecisionTreeClassifier(Classifier):
    '''
    For sklearn's sklearn.tree.DecisionTreeClassifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier
    '''

    #########################################
    @staticmethod
    def __MAKE_MODEL(max_depth, min_samples_leaf):
        '''Make an sklearn model from parameters.'''
        return sklearn.pipeline.Pipeline([
            (
                'preprocessor',
                sklearn.preprocessing.StandardScaler(with_mean=False, with_std=False)
                ),
            (
                'classifier',
                sklearn.tree.DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight='balanced',
                    random_state=0
                    )
                )
            ])

    #########################################
    def __init__(self, labels, max_depth, min_samples_leaf, sklearn_model=None):
        '''
        Constructor.

        :param list labels: List of labels to be classified.
        :param max_depth: The maximum depth of each tree.
        :type max_depth: int or samplers.Sampler
        :param min_samples_leaf: The minimum number of items per leaf.
        :type min_samples_leaf: int or samplers.Sampler
        :param sklearn_model: The pretrained sklearn model to use, if any.
            If None then an untrained model will be created. Otherwise
            it is validated against the given parameters.
        :type sklearn_model: None or sklearn_DecisionTreeClassifier
        '''
        super().__init__(
            labels,
            (
                sklearn_model
                if sklearn_model is not None
                else self.__MAKE_MODEL(max_depth, min_samples_leaf)
                if (
                    not isinstance(max_depth, samplers.Sampler)
                    and not isinstance(min_samples_leaf, samplers.Sampler)
                    )
                else None
                )
            )

        self.max_depth = None
        self.min_samples_leaf = None
        self.max_depth_sampler = None
        self.min_samples_leaf_sampler = None
        if isinstance(max_depth, samplers.Sampler):
            self.max_depth_sampler = max_depth
        else:
            self.max_depth = max_depth
        if isinstance(min_samples_leaf, samplers.Sampler):
            self.min_samples_leaf_sampler = min_samples_leaf
        else:
            self.min_samples_leaf = min_samples_leaf

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values and resulting sklearn model from the samplers provided.
        '''
        self.max_depth = self.max_depth_sampler.get_value()
        self.min_samples_leaf = self.min_samples_leaf_sampler.get_value()

        self.sklearn_model = self.__MAKE_MODEL(self.max_depth, self.min_samples_leaf)

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the classifier parameters.
        '''
        self.max_depth_sampler.set_value(config['params']['max_depth'])
        self.min_samples_leaf_sampler.set_value(config['params']['min_samples_leaf'])

    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the classifier's parameters.

        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {
            'type': 'decision_tree',
            'params': {
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the classifier's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (self.max_depth, self.min_samples_leaf)

    #########################################
    def get_picklable(self):
        '''
        Get the classifier's inner model in a picklable form.

        :return: The picklable object.
        :rtype: object
        '''
        return self.sklearn_model


#########################################
class RandomForestClassifier(Classifier):
    '''
    For sklearn's sklearn.ensemble.RandomForestClassifier.

    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier
    '''

    #########################################
    @staticmethod
    def __MAKE_MODEL(n_estimators, max_depth, min_samples_leaf):
        '''Make an sklearn model from parameters.'''
        return sklearn.pipeline.Pipeline([
            (
                'preprocessor',
                sklearn.preprocessing.StandardScaler(with_mean=False, with_std=False)
                ),
            (
                'classifier',
                sklearn.ensemble.RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight='balanced',
                    verbose=0,
                    random_state=0
                    )
                )
            ])

    #########################################
    def __init__(self, labels, n_estimators, max_depth, min_samples_leaf, sklearn_model=None):
        '''
        Constructor.

        :param list labels: List of labels to be classified.
        :param n_estimators: The number of trees in the forest.
        :type n_estimators: int or samplers.Sampler
        :param max_depth: The maximum depth of each tree.
        :type max_depth: int or samplers.Sampler
        :param min_samples_leaf: The minimum number of items per leaf.
        :type min_samples_leaf: int or samplers.Sampler
        :param sklearn_model: The pretrained sklearn model to use, if any.
            If None then an untrained model will be created. Otherwise
            it is validated against the given parameters.
        :type sklearn_model: None or sklearn_RandomForestClassifier
        '''
        super().__init__(
            labels,
            (
                sklearn_model
                if sklearn_model is not None
                else self.__MAKE_MODEL(n_estimators, max_depth, min_samples_leaf)
                if (
                    not isinstance(n_estimators, samplers.Sampler)
                    and not isinstance(max_depth, samplers.Sampler)
                    and not isinstance(min_samples_leaf, samplers.Sampler)
                    )
                else None
                )
            )

        self.n_estimators = None
        self.max_depth = None
        self.min_samples_leaf = None
        self.n_estimators_sampler = None
        self.max_depth_sampler = None
        self.min_samples_leaf_sampler = None
        if isinstance(n_estimators, samplers.Sampler):
            self.n_estimators_sampler = n_estimators
        else:
            self.n_estimators = n_estimators
        if isinstance(max_depth, samplers.Sampler):
            self.max_depth_sampler = max_depth
        else:
            self.max_depth = max_depth
        if isinstance(min_samples_leaf, samplers.Sampler):
            self.min_samples_leaf_sampler = min_samples_leaf
        else:
            self.min_samples_leaf = min_samples_leaf

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values and resulting sklearn model from the samplers provided.
        '''
        self.n_estimators = self.n_estimators_sampler.get_value()
        self.max_depth = self.max_depth_sampler.get_value()
        self.min_samples_leaf = self.min_samples_leaf_sampler.get_value()

        self.sklearn_model = self.__MAKE_MODEL(self.n_estimators, self.max_depth, self.min_samples_leaf)

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the classifier parameters.
        '''
        self.n_estimators_sampler.set_value(config['params']['n_estimators'])
        self.max_depth_sampler.set_value(config['params']['max_depth'])
        self.min_samples_leaf_sampler.set_value(config['params']['min_samples_leaf'])

    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the classifier's parameters.

        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {
            'type': 'random_forest',
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_leaf': self.min_samples_leaf
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the classifier's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (self.n_estimators, self.max_depth, self.min_samples_leaf)


    #########################################
    def get_picklable(self):
        '''
        Get the classifier's inner model in a picklable form.

        :return: The picklable object.
        :rtype: object
        '''
        return self.sklearn_model


#########################################
class TensorflowNeuralNetworkClassifier(Classifier):
    '''
    A custom made neural network that uses Tensorflow.
    '''

    #########################################
    @staticmethod
    def __MAKE_MODEL(hidden_layer_sizes, dropout_rate, init_stddev, batch_size, max_iter, max_batch_memory, use_gpu):
        '''Make an sklearn model from parameters.'''
        return sklearn.pipeline.Pipeline([
            (
                'preprocessor',
                sklearn.preprocessing.StandardScaler()
                ),
            (
                'classifier',
                SklearnLikeTensorflowNeuralNet(
                    hidden_layer_sizes=hidden_layer_sizes,
                    dropout_rate=dropout_rate,
                    init_stddev=init_stddev,
                    validation_fraction=0.1,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    patience=3,
                    verbose=False,
                    random_state=0,
                    max_batch_memory=max_batch_memory,
                    use_gpu=use_gpu
                    )
                )
            ])

    #########################################
    def __init__(self, labels, hidden_layer_sizes, dropout_rate, init_stddev, batch_size, max_iter, max_batch_memory, use_gpu, sklearn_model=None):
        '''
        Constructor.

        :param list labels: List of labels to be classified.
        :param hidden_layer_sizes: The amount of neural units in each
            hidden layer.
        :type hidden_layer_sizes: list or samplers.Sampler
        :param dropout: The amount to regularise the model such that
            0.5 is the greatest amount and 0.0 is no regularisation.
        :type dropout: float or samplers.Sampler
        :param init_stddev: The standard deviation of the random normal number
            generator to use (with mean 0) for initialising the weights.
        :type dropout: float or samplers.Sampler
        :param batch_size: The number of training items in each
            minibatch.
        :type batch_size: int or samplers.Sampler
        :param max_iter: The number of iterations to spend on
            training.
        :type max_iter: int or samplers.Sampler
        :param float max_batch_memory: The maximum number of gigabytes of memory to use.
        :param bool use_gpu: Whether to use the GPU.
        :param sklearn_model: The pretrained sklearn model to use, if any.
            If None then an untrained model will be created. Otherwise
            it is validated against the given parameters.
        :type sklearn_model: None or sklearn_LogisticRegression
        '''
        super().__init__(
            labels,
            (
                sklearn.pipeline.Pipeline([
                    ('preprocessor', sklearn_model['preprocessor']),
                    ('classifier', SklearnLikeTensorflowNeuralNet.load_from_pickle(sklearn_model['classifier'], max_batch_memory, use_gpu))
                    ])
                if sklearn_model is not None
                else self.__MAKE_MODEL(hidden_layer_sizes, dropout_rate, init_stddev, batch_size, max_iter, max_batch_memory, use_gpu)
                if (
                    not any(isinstance(hidden_layer_sizes, samplers.Sampler) for hidden_layer_size in hidden_layer_sizes)
                    and not isinstance(dropout_rate, samplers.Sampler)
                    and not isinstance(init_stddev, samplers.Sampler)
                    and not isinstance(batch_size, samplers.Sampler)
                    and not isinstance(max_iter, samplers.Sampler)
                    )
                else None
                )
            )

        self.hidden_layer_sizes = [None]*len(hidden_layer_sizes)
        self.dropout_rate = None
        self.init_stddev = None
        self.batch_size = None
        self.max_iter = None
        self.hidden_layer_sizes_samplers = [None]*len(hidden_layer_sizes)
        self.dropout_rate_sampler = None
        self.init_stddev_sampler = None
        self.batch_size_sampler = None
        self.max_iter_sampler = None
        for i in range(len(hidden_layer_sizes)):
            if isinstance(hidden_layer_sizes[i], samplers.Sampler):
                self.hidden_layer_sizes_samplers[i] = hidden_layer_sizes[i]
            else:
                self.hidden_layer_sizes[i] = hidden_layer_sizes[i]
        if isinstance(dropout_rate, samplers.Sampler):
            self.dropout_rate_sampler = dropout_rate
        else:
            self.dropout_rate = dropout_rate
        if isinstance(init_stddev, samplers.Sampler):
            self.init_stddev_sampler = init_stddev
        else:
            self.init_stddev = init_stddev
        if isinstance(batch_size, samplers.Sampler):
            self.batch_size_sampler = batch_size
        else:
            self.batch_size = batch_size
        if isinstance(max_iter, samplers.Sampler):
            self.max_iter_sampler = max_iter
        else:
            self.max_iter = max_iter
        self.max_batch_memory = max_batch_memory
        self.use_gpu = use_gpu

    #########################################
    def refresh_parameters(self):
        '''
        Refresh parameter values and resulting sklearn model from the samplers provided.
        '''
        for i in range(len(self.hidden_layer_sizes)):
            self.hidden_layer_sizes[i] = self.hidden_layer_sizes_samplers[i].get_value()
        self.dropout_rate = self.dropout_rate_sampler.get_value()
        self.init_stddev = self.init_stddev_sampler.get_value()
        self.batch_size = self.batch_size_sampler.get_value()
        self.max_iter = self.max_iter_sampler.get_value()

        self.sklearn_model = self.__MAKE_MODEL(self.hidden_layer_sizes, self.dropout_rate, self.init_stddev, self.batch_size, self.max_iter, self.max_batch_memory, self.use_gpu)

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the classifier parameters.
        '''
        for i in range(len(self.hidden_layer_sizes)):
            self.hidden_layer_sizes_samplers[i].set_value(config['params']['hidden_layer_sizes'][i])
        self.dropout_rate_sampler.set_value(config['params']['dropout_rate'])
        self.init_stddev_sampler.set_value(config['params']['init_stddev'])
        self.batch_size_sampler.set_value(config['params']['batch_size'])
        self.max_iter_sampler.set_value(config['params']['max_iter'])

    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of the classifier's parameters.

        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {
            'type': 'tensorflow_neural_network',
            'params': {
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'dropout_rate': self.dropout_rate,
                'init_stddev': self.init_stddev,
                'batch_size': self.batch_size,
                'max_iter': self.max_iter
                }
            }

    #########################################
    def get_params(self):
        '''
        Get the classifier's parameters as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (tuple(self.hidden_layer_sizes), self.dropout_rate, self.init_stddev, self.batch_size, self.max_iter)

    #########################################
    def get_picklable(self):
        '''
        Get the classifier's inner model in a picklable form.

        :return: The picklable object.
        :rtype: object
        '''
        return {
            'preprocessor': self.sklearn_model.named_steps['preprocessor'],
            'classifier': self.sklearn_model.named_steps['classifier'].get_picklable()
            }
