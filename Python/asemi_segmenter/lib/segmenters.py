'''Module that puts together segmentation related processes.'''

import pickle
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import classifiers
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import samplers


#########################################
def load_segmenter_from_pickle_data(pickle_data, full_volume, use_gpu=False):
    '''
    Load a segmenter from pickled data.

    :param dict pickle_data: The loaded contents of a segmenter pickle.
    :param FullVolume full_volume: The full volume object containing the voxels to work on.
    :param bool use_gpu: Whether to use the GPU to compute features.
    :return: A loaded segmenter object.
    :rtype: Segmenter
    '''
    if not isinstance(pickle_data['labels'], list):
        raise ValueError('Pickle is invalid as labels are not a list.')
    for (i, entry) in enumerate(pickle_data['labels']):
        if not isinstance(entry, str):
            raise ValueError('Pickle is invalid as label entry {} is not a string.'.format(i))

    return Segmenter(
        pickle_data['labels'],
        full_volume,
        pickle_data['config'],
        sklearn_model=pickle_data['sklearn_model'],
        use_gpu=use_gpu
        )


#########################################
class Segmenter(object):
    '''An object that puts together everything needed to segment a volume after it has been processed.'''

    #########################################
    def __init__(self, labels, full_volume, train_config, sklearn_model=None, sampler_factory=None, use_gpu=False):
        '''
        Constructor.

        :param list labels: List of labels for the classifier.
        :param FullVolume full_volume: Full volume on which the segmenter will be working.
        :param dict train_config: Loaded configuration of the training method.
        :param sklearn_model sklearn_model: sklearn model to use for machine learning, if
            pretrained.
        :param samplers.SamplerFactory sampler_factory: The factory to create samplers
            to randomly generate parameters.
        :param bool use_gpu: Whether to use the GPU to compute features.
        '''
        self.sampler_factory = sampler_factory

        validations.validate_json_with_schema_file(train_config, 'train.json')
        featuriser = featurisers.load_featuriser_from_config(train_config['featuriser'], self.sampler_factory, use_gpu)
        classifier = classifiers.load_classifier_from_config(labels, train_config['classifier'], sklearn_model, self.sampler_factory)

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
    def refresh_params(self):
        '''
        Refresh parameters of featuriser and classifier from samplers.
        '''
        self.featuriser.refresh_parameters()
        self.classifier.refresh_parameters()

        scales_needed = self.featuriser.get_scales_needed()
        if not scales_needed <= self.full_volume.get_scales():
            raise ValueError(
                'Featuriser requires scales that are not included in preprocessed volume ' \
                '(missing scales=[{}]).'.format(
                    ', '.join(sorted(scales_needed - self.full_volume.get_scales()))
                    )
                )

    #########################################
    def set_sampler_values(self, config):
        '''
        Set the values of the samplers provided according to a config.

        :param dict config: The configuration dictionary for the training parameters.
        '''
        self.featuriser.set_sampler_values(config['featuriser'])
        self.classifier.set_sampler_values(config['classifier'])

    #########################################
    def train(self, training_set, max_processes=1, verbose_training=False):
        '''
        Train the classifier using a provided training set.

        The training set is to be constructed from voxels in the full volume specified in
        the constructor.

        :param TrainingSet training_set: Training set of feature/label pairs to train the
            classifier where the features were constructed using the featuriser in this
            object.
        :param int max_processes: Number of processes to use.
        :param bool verbose_training: Whether to include sklearn's verbose texts.
        '''
        self.classifier.train(training_set, max_processes, verbose_training)

    #########################################
    def segment_to_label_indexes(self, features, max_processes=1):
        '''
        Classify an array of feature vectors into the most probable label.

        :param numpy.ndarray features: 1D numpy array of features representing voxels to segment.
            These features are expected to be constructed using the featuriser in this object.
        :param int max_processes: Number of processes to use.
        :return: A 1D numpy array of indexes where each index is a label in the this object's
            classifier's labels.
        :rtype: numpy.ndarray
        '''
        return self.classifier.predict_label_indexes(features, max_processes)

    #########################################
    def segment_to_labels_iter(self, features, soft_segmentation=False, max_processes=1):
        '''
        Classify an array of feature vectors into a binary mask for each label.

        Each mask is returned separately in a generator.

        :param numpy.ndarray features: 1D numpy array of features representing voxels to segment.
            These features are expected to be constructed using the featuriser in this object.
        :param bool soft_segmentation: Whether the binary mask should use probabilities instead of
            a hard 1 or 0 with only one label being 1 for each voxel.
        :param int max_processes: Number of processes to use.
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
    def get_params(self):
        '''
        Get the parameters of the featuriser and classifier as nested tuples.

        :return: The parameters.
        :rtype: tuple
        '''
        return (self.featuriser.get_params(), self.classifier.get_params())

    #########################################
    def save(self, fname):
        '''
        Save the segmenter data to a pickle file.

        Pickle uses protocol 2.

        :param str fname: The full file name (with path) of the pickle file.
        '''
        pickle_data = {
            'sklearn_model': self.classifier.get_picklable(),
            'labels': self.classifier.labels,
            'config': self.get_config()
            }
        with open(fname, 'wb') as f:
            pickle.dump(pickle_data, f, protocol=2)