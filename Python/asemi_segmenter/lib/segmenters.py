'''Module that puts together segmentation related processes.'''

import pickle
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import classifiers
from asemi_segmenter.lib import validations


#########################################
def load_segmenter_from_pickle_data(pickle_data, full_volume, allow_random=False):
    '''
    Load a segmenter from pickled data.
    
    :param dict pickle_data: The loaded contents of a segmenter pickle.
    :param FullVolume full_volume: The full volume object containing the voxels to work on.
    :param bool allow_random: Whether to allow configurations that specify how to randomly generate parameters.
    :return: A loaded segmenter object.
    :rtype: Segmenter
    '''
    if not isinstance(pickle_data['labels'], list):
        raise ValueError('Pickle is invalid as labels are not a list.')
    for (i, entry) in enumerate(pickle_data['labels']):
        if not isinstance(entry, str):
            raise ValueError('Pickle is invalid as label entry {} is not a string.'.format(i))
    
    return Segmenter(pickle_data['labels'], full_volume, pickle_data['config'], pickle_data['model'], allow_random)


#########################################
class Segmenter(object):
    '''An object that puts together everything needed to segment a volume after it has been processed.'''

    #########################################
    def __init__(self, labels, full_volume, train_config, model=None, allow_random=False):
        '''
        Constructor.
        
        :param list labels: List of labels for the classifier.
        :param FullVolume full_volume: Full volume on which the segmenter will be working.
        :param dict train_config: Loaded configuration of the training method.
        :param sklearn_model model: sklearn model to use for machine learning, if pretrained.
        :param bool allow_random: Whether to allow training configurations that specify how to randomly generate parameters.
        '''
        validations.validate_json_with_schema_file(train_config, 'train.json')
        featuriser = featurisers.load_featuriser_from_config(train_config['featuriser'], allow_random)
        classifier = classifiers.load_classifier_from_config(labels, train_config['classifier'], model, allow_random)
        
        if train_config['training_set']['sample_size_per_label'] == 0:
            raise ValueError('Model is invalid as sample_size_per_label cannot be 0.')
            
        if model is not None:
            if model.n_features_ != featuriser.get_feature_size():
                raise ValueError(
                    'Model is invalid as the number of features is not as declared (declared={}, ' \
                    'actual={}).'.format(
                        featuriser.get_feature_size(),
                        model.n_features_
                        )
                    )

        scales_needed = featuriser.get_scales_needed()
        if None not in scales_needed and not scales_needed <= full_volume.get_scales():
            raise ValueError(
                'Featuriser requires scales that are not included in preprocessed volume ' \
                '(missing scales=[{}]).'.format(
                    ', '.join(sorted(scales_needed - full_volume.get_scales()))
                    )
                )

        self.train_config = train_config
        self.featuriser = featuriser
        self.classifier = classifier
        self.full_volume = full_volume
    
    #########################################
    def regenerate(self):
        '''
        Regenerate parameters with value generators provided.
        '''
        self.featuriser.regenerate()
        self.classifier.regenerate()
        
        scales_needed = self.featuriser.get_scales_needed()
        if not scales_needed <= self.full_volume.get_scales():
            raise ValueError(
                'Featuriser requires scales that are not included in preprocessed volume ' \
                '(missing scales=[{}]).'.format(
                    ', '.join(sorted(scales_needed - self.full_volume.get_scales()))
                    )
                )
    
    #########################################
    def train(self, training_set, n_jobs=1):
        '''
        Train the classifier using a provided training set.
        
        The training set is to be constructed from voxels in the full volume specified in
        the constructor.
        
        :param TrainingSet training_set: Training set of feature/label pairs to train the
            classifier where the features were constructed using the featuriser in this
            object.
        :param int n_jobs: Number of processes to use.
        '''
        sample_size_per_label = self.train_config['training_set']['sample_size_per_label']
        new_training_set = None
        if sample_size_per_label == -1:
            new_training_set = training_set.without_control_labels()
        else:
            new_training_set = training_set.get_sample(sample_size_per_label, seed=0)
        self.classifier.train(new_training_set, n_jobs)
    
    #########################################
    def segment_to_label_indexes(self, features, n_jobs=1):
        '''
        Classify an array of feature vectors into the most probable label.
        
        :param numpy.ndarray features: 1D numpy array of features representing voxels to segment.
            These features are expected to be constructed using the featuriser in this object.
        :param int n_jobs: Number of processes to use.
        :return: A 1D numpy array of indexes where each index is a label in the this object's
            classifier's labels.
        :rtype: numpy.ndarray
        '''
        return self.classifier.predict_label_indexes(features, n_jobs)
    
    #########################################
    def segment_to_labels_iter(self, features, soft_segmentation=False, n_jobs=1):
        '''
        Classify an array of feature vectors into a binary mask for each label.
        
        Each mask is returned separately in a generator.
        
        :param numpy.ndarray features: 1D numpy array of features representing voxels to segment.
            These features are expected to be constructed using the featuriser in this object.
        :param bool soft_segmentation: Whether the binary mask should use probabilities instead of
            a hard 1 or 0 with only one label being 1 for each voxel.
        :param int n_jobs: Number of processes to use.
        :return: A generator with a 1D numpy array of probabilities or binary values specifying how
            much a corresponding feature vector (voxel) belongs to the current label where each
            iteration in the generator is a corresponding label in the this object's classifier's
            labels.
        :rtype: iter
        '''
        if soft_segmentation:
            vectors = self.classifier.predict_label_probs(features)
        else:
            vectors = self.classifier.predict_label_onehots(features)
        
        for i in range(len(self.classifier.labels)):
            yield vectors[:, i]
    
    #########################################
    def get_config(self):
        '''
        Get the dictionary configuration of this segmenter's training configuration.
        
        :return: The dictionary configuration.
        :rtype: dict
        '''
        return {
            'featuriser': self.featuriser.get_config(),
            'classifier': self.classifier.get_config(),
            'training_set': self.train_config['training_set']
            }
    
    #########################################
    def save(self, fname):
        '''
        Save the model to a pickle file.

        Pickle uses protocol 2.

        :param str fname: The full file name (with path) of the pickle file.
        '''
        pickle_data = {
            'model': self.classifier.model,
            'labels': self.classifier.labels,
            'config': self.get_config()
            }
        with open(fname, 'wb') as f:
            pickle.dump(pickle_data, f, protocol=2)