'''Module with functions related to saving results iteratively.'''

import json
import numpy as np


#########################################
class EvaluationResultsFile(object):
    '''Results file interface for the evaluate command.'''

    #########################################
    def __init__(self, results_fullfname):
        '''
        Create an evaluation results file object.

        :param results_fullfname: The full file name (with path) of the results text file. If
            None then no file will be saved and all inputs are ignored.
        :type results_fullfname: str or None
        '''
        self.results_fullfname = results_fullfname

    #########################################
    def create(self, labels):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'slice', *labels, 'featurisation duration (s)', 'prediction duration (s)',
                    sep='\t', file=f
                    )

    #########################################
    def append(self, slice_fullfname, ious, featuriser_duration, classifier_duration):
        '''
        Add a new slice's result to the file.

        :param str slice_fullfname: The full file name of the slice being used for evaluation.
        :param list ious: The list of intersection-over-union scores for each label.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    slice_fullfname,
                    *[('{:.3%}'.format(iou) if iou is not None else '') for iou in ious],
                    '{:.1f}'.format(featuriser_duration),
                    '{:.1f}'.format(classifier_duration),
                    sep='\t', file=f
                    )


#########################################
class TuningResultsFile(object):
    '''Results file interface for the tune command.'''

    #########################################
    def __init__(self, results_fullfname):
        '''
        Create a tune results file object.

        :param results_fullfname: The full file name (with path) of the results text file. If
            None then no file will be saved and all inputs are ignored.
        :type results_fullfname: str or None
        '''
        self.results_fullfname = results_fullfname

    #########################################
    def create(self, labels):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'json_config', *['{}_iou'.format(label) for label in labels], 'mean_iou', 'min_iou',
                    sep='\t', file=f
                    )

    #########################################
    def append(self, config, ious):
        '''
        Add a new slice's result to the file.

        :param dict config: The configuation dictionary used to produce these results.
        :param list ious: The list of intersection-over-union scores for each label.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    json.dumps(config),
                    *['{:.3%}'.format(iou) for iou in ious],
                    '{:.3%}'.format(np.mean(ious).tolist()),
                    '{:.3%}'.format(np.min(ious).tolist()),
                    sep='\t', file=f
                    )