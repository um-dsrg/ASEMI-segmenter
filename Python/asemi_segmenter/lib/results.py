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
        self.evaluation = None
        self.total_featuriser_duration = 0.0
        self.total_classifier_duration = 0.0
        self.num_rows = 0

    #########################################
    def create(self, labels, evaluation):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        :param Evaluation evaluation: The evaluation object being used for the results.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'slice',
                    *['{} {}'.format(label, evaluation.name) for label in labels],
                    'global {}'.format(evaluation.name),
                    'min {}'.format(evaluation.name),
                    'stddev {}'.format(evaluation.name),
                    'featurisation duration (s)',
                    'prediction duration (s)',
                    sep='\t', file=f
                    )
        self.evaluation = evaluation

    #########################################
    def add(self, slice_fullfname, label_results, global_result, featuriser_duration, classifier_duration):
        '''
        Add a new slice's result to the file.

        :param str slice_fullfname: The full file name of the slice being used for evaluation.
        :param list label_results: The list of scores for each label.
        :param float global_result: The global score.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    slice_fullfname,
                    *[
                        (
                            '{:.3{}}'.format(
                                result,
                                '%' if self.evaluation.is_percentage else 'f'
                                )
                            if result is not None else ''
                            )
                        for result in label_results
                        ],
                    '{:.3{}}'.format(
                        global_result,
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        min(label_results),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        np.std(label_results).tolist(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.1f}'.format(featuriser_duration),
                    '{:.1f}'.format(classifier_duration),
                    sep='\t', file=f
                    )
        self.total_featuriser_duration += featuriser_duration
        self.total_classifier_duration += classifier_duration
        self.num_rows += 1
    
    #########################################
    def conclude(self):
        '''
        Write the last row with the global results.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    'global',
                    *[
                        (
                            '{:.3{}}'.format(
                                result,
                                '%' if self.evaluation.is_percentage else 'f'
                                )
                            if result is not None else ''
                            )
                        for result in self.evaluation.get_global_result_per_label()
                        ],
                    '{:.3{}}'.format(
                        self.evaluation.get_global_result(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        min(self.evaluation.get_global_result_per_label()),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        np.std(self.evaluation.get_global_result_per_label()).tolist(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.1f}'.format(self.total_featuriser_duration/self.num_rows),
                    '{:.1f}'.format(self.total_classifier_duration/self.num_rows),
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
        self.evaluation = None

    #########################################
    def create(self, labels, evaluation):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        :param Evaluation evaluation: The evaluation object being used for the results.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'json config',
                    *['{} {}'.format(label, evaluation.name) for label in labels],
                    'global {}'.format(evaluation.name),
                    'min {}'.format(evaluation.name),
                    'stddev {}'.format(evaluation.name),
                    'featuriser duration (s)',
                    'classifier duration (s)',
                    'max memory (MB)',
                    sep='\t', file=f
                    )
        self.evaluation = evaluation

    #########################################
    def add(self, config, featuriser_duration, classifier_duration, max_memory_mb):
        '''
        Add a new result to the file.

        :param dict config: The configuation dictionary used to produce these results.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        :param float max_memory_mb: The maximum number of megabytes of memory used
            during featurisation and classification.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    json.dumps(config),
                    *[
                        (
                            '{:.3{}}'.format(
                                result,
                                '%' if self.evaluation.is_percentage else 'f'
                                )
                            if result is not None else ''
                            )
                        for result in self.evaluation.get_global_result_per_label()
                        ],
                    '{:.3{}}'.format(
                        self.evaluation.get_global_result(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        min(self.evaluation.get_global_result_per_label()),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        np.std(self.evaluation.get_global_result_per_label()).tolist(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.1f}'.format(featuriser_duration),
                    '{:.1f}'.format(classifier_duration),
                    '{:.3f}'.format(max_memory_mb),
                    sep='\t', file=f
                    )