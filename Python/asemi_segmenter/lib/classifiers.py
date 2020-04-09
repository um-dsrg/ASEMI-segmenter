'''Module containing classifiers to classify voxels into labels.'''

import random
import numpy as np
import sklearn.ensemble
from asemi_segmenter.lib import validations


#########################################
def load_classifier_from_config(labels, config, model=None, allow_random=False):
    '''
    Load a classifier from a configuration dictionary.
    
    :param list labels: List of labels to recognise.
    :param dict config: Configuration of the classifier.
    :param sklearn_model model: Trained model if available.
    :return: A classifier object.
    :rtype: Classifier
    '''
    validations.validate_json_with_schema_file(config, 'classifier.json')
    rand = random.Random(0)
    
    if config['type'] == 'random_forest':
        n_estimators = None
        max_depth = None
        min_samples_leaf = None
        
        if isinstance(config['params']['n_estimators'], dict):
            if not allow_random:
                raise ValueError('n_estimators must be a constant not a range.')
            if (config['params']['n_estimators']['min'] >= config['params']['n_estimators']['max']):
                raise ValueError('n_estimators min is not less than n_estimators max.')
            n_estimators = lambda:rand.randrange(
                config['params']['n_estimators']['min'],
                config['params']['n_estimators']['max'] + 1
                )
        else:
            n_estimators = config['params']['n_estimators']
        
        if isinstance(config['params']['max_depth'], dict):
            if not allow_random:
                raise ValueError('max_depth must be a constant not a range.')
            if (config['params']['max_depth']['min'] >= config['params']['max_depth']['max']):
                raise ValueError('max_depth min is not less than max_depth max.')
            max_depth = lambda:rand.randrange(
                config['params']['max_depth']['min'],
                config['params']['max_depth']['max'] + 1
                )
        else:
            max_depth = config['params']['max_depth']
        
        if isinstance(config['params']['min_samples_leaf'], dict):
            if not allow_random:
                raise ValueError('min_samples_leaf must be a constant not a range.')
            if (config['params']['min_samples_leaf']['min'] >= config['params']['min_samples_leaf']['max']):
                raise ValueError('min_samples_leaf min is not less than min_samples_leaf max.')
            min_samples_leaf = lambda:rand.randrange(
                config['params']['min_samples_leaf']['min'],
                config['params']['min_samples_leaf']['max'] + 1
                )
        else:
            min_samples_leaf = config['params']['min_samples_leaf']
        
        if model is not None:
            if not isinstance(model, sklearn.ensemble.RandomForestClassifier):
                raise ValueError('Model is invalid as it is not a random forest as declared.')
            if model.n_classes_ != len(labels):
                raise ValueError(
                    'Model is invalid as the number of classes is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        len(labels),
                        model.n_classes_
                        )
                    )
            if model.n_estimators != n_estimators:
                raise ValueError('Model is invalid as n_estimators is not as declared.')
            if model.max_depth != max_depth:
                raise ValueError('Model is invalid as max_depth is not as declared.')
            if model.min_samples_leaf != min_samples_leaf:
                raise ValueError('Model is invalid as min_samples_leaf is not as declared.')
        
        return RandomForestClassifier(labels, n_estimators, max_depth, min_samples_leaf, model)


#########################################
class Classifier(object):
    '''Super class for classifiers.'''
    
    #########################################
    def __init__(self, labels, model=None):
        '''
        Constructor.
        
        :param list labels: List of labels to be classified.
        :param sklearn_model model: The sklearn model to use as a classification.
            If None then model is expected to be created later in the regenerate
            method.
        '''
        self.labels = labels
        self.model = model
    
    #########################################
    def regenerate(self):
        '''
        Regenerate parameters and resulting model with value generators provided.
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
    def train(self, training_set, n_jobs=1):
        '''
        Turn a slice from a volume into a matrix of feature vectors.

        :param int n_jobs: The number of concurrent processes to use.
        :return: A reference to output.
        :rtype: numpy.ndarray
        '''
        self.model.n_jobs = n_jobs
        self.model.fit(
            training_set.get_features_array(),
            training_set.get_labels_array()
            )
    
    #########################################
    def predict_label_probs(self, features_array, n_jobs=1):
        '''
        Predict the probability of each label for each feature vector.
        
        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int n_jobs: The number of concurrent processes to use.
        :return: 2D numpy array with each row being the probabilities for the corresponding
            feature vector and each column being a label.
        :rtype: numpy.ndarray
        '''
        self.model.n_jobs = n_jobs
        return self.model.predict_proba(features_array)
    
    #########################################
    def predict_label_onehots(self, features_array, n_jobs=1):
        '''
        Predict one hot vectors of each label for each feature vector.
        
        The label with the highest probability will get a 1 in the vectors' corresponding index
        and the other vector elements will be 0.
        
        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int n_jobs: The number of concurrent processes to use.
        :return: 2D numpy array with each row being the one hot vectors for the corresponding
            feature vector and each column being a label.
        :rtype: numpy.ndarray
        '''
        probs = self.predict_label_probs(features_array, n_jobs)
        label_indexes = np.argmax(probs, axis=1)
        probs[:, :] = 0.0
        probs[np.arange(probs.shape[0]), label_indexes] = 1.0
        return probs
    
    #########################################
    def predict_label_indexes(self, features_array, n_jobs=1):
        '''
        Predict a label index for each feature vector.
        
        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int n_jobs: The number of concurrent processes to use.
        :return: An array of integers with each item being the label index for the corresponding feature vector.
        :rtype: numpy.ndarray
        '''
        probs = self.predict_label_probs(features_array, n_jobs)
        return np.argmax(probs, axis=1)
    
    #########################################
    def predict_label_names(self, features_array, n_jobs=1):
        '''
        Predict a label name for each feature vector.
        
        :param numpy.ndarray features_array: 2D numpy array with each row being a feature vector
            to pass to the classifier.
        :param int n_jobs: The number of concurrent processes to use.
        :return: A list of string with each item being the label name for the corresponding feature vector.
        :rtype: list
        '''
        label_indexes = self.predict_label_indexes(features_array, n_jobs)
        return [self.labels[label_index] for label_index in label_indexes.tolist()]
    
    
#########################################
class RandomForestClassifier(Classifier):
    '''For sklearn's RandomForestClassifier.'''
    
    #########################################
    def __init__(self, labels, n_estimators, max_depth, min_samples_leaf, model=None):
        '''
        Constructor.
        
        :param list labels: List of labels to be classified.
        :param n_estimators: The number of trees in the forest.
        :type n_estimators: int or callable
        :param max_depth: The maximum depth of each tree.
        :type max_depth: int or callable
        :param min_samples_leaf: The minimum number of items per leaf.
        :type min_samples_leaf: int or callable
        :param model: The minimum number of items per leaf.
        :type model: None or sklearn_RandomForestClassifier
        '''
        super().__init__(
            labels,
            (
                model
                if model is not None
                else sklearn.ensemble.RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=0
                    )
                if (
                    n_estimators is not None
                    and max_depth is not None
                    and min_samples_leaf is not None
                    )
                else None
                )
            )
        
        self.n_estimators = n_estimators if isinstance(n_estimators, int) else None
        self.max_depth = max_depth if isinstance(max_depth, int) else None
        self.min_samples_leaf = min_samples_leaf if isinstance(min_samples_leaf, int) else None
        
        if isinstance(n_estimators, int):
            self.n_estimators_generator = lambda:n_estimators
        elif callable(n_estimators):
            self.n_estimators_generator = n_estimators
        else:
            raise ValueError('n_estimators must be int or callable.')
        
        if isinstance(max_depth, int):
            self.max_depth_generator = lambda:max_depth
        elif callable(max_depth):
            self.max_depth_generator = max_depth
        else:
            raise ValueError('max_depth must be int or callable.')
        
        if isinstance(min_samples_leaf, int):
            self.min_samples_leaf_generator = lambda:min_samples_leaf
        elif callable(min_samples_leaf):
            self.min_samples_leaf_generator = min_samples_leaf
        else:
            raise ValueError('min_samples_leaf must be int or callable.')
    
    #########################################
    def regenerate(self):
        '''
        Regenerate parameters and resulting model with value generators provided.
        '''
        self.n_estimators = self.n_estimators_generator()
        self.max_depth = self.max_depth_generator()
        self.min_samples_leaf = self.min_samples_leaf_generator()
        self.model = sklearn.ensemble.RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0
            )
    
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