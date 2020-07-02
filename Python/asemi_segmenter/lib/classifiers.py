'''Module containing classifiers to classify voxels into labels.'''

import random
import warnings
import numpy as np
import sklearn
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neural_network
import sklearn.tree
import sklearn.ensemble
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import samplers
from asemi_segmenter.lib import redirectors


#########################################
def load_classifier_from_config(labels, config, sklearn_model=None, sampler_factory=None):
    '''
    Load a classifier from a configuration dictionary.

    :param list labels: List of labels to recognise.
    :param dict config: Configuration of the classifier.
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

    else:
        raise NotImplementedError('Classifier {} not implemented.'.format(config['type']))


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
    def train(self, training_set, max_processes=1, listener=None):
        '''
        Turn a slice from a volume into a matrix of feature vectors.

        :param int max_processes: The number of concurrent processes to use.
        :param callable listener: The listener to receive the sklearn verbose texts.
            Listener should accept one string argument.
        :return: A reference to output.
        :rtype: numpy.ndarray
        '''
        self.sklearn_model.named_steps['classifier'].max_processes = max_processes

        if listener is not None and hasattr(self.sklearn_model.named_steps['classifier'], 'verbose'):
            if isinstance(self.sklearn_model.named_steps['classifier'].verbose, int):
                self.sklearn_model.named_steps['classifier'].verbose = 2
            elif isinstance(self.sklearn_model.named_steps['classifier'].verbose, bool):
                self.sklearn_model.named_steps['classifier'].verbose = True

        with redirectors.PrintRedirector(listener):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=sklearn.exceptions.ConvergenceWarning)

                self.sklearn_model.fit(
                    training_set.get_features_array(),
                    training_set.get_labels_array()
                    )

        if listener is not None and hasattr(self.sklearn_model.named_steps['classifier'], 'verbose'):
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
        self.sklearn_model.named_steps['classifier'].max_processes = max_processes
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