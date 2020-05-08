'''Module with functions related to saving results.'''

import json
import numpy as np
from asemi_segmenter.lib import colours
from asemi_segmenter.lib import images
from asemi_segmenter.lib import evaluations


#########################################
def save_confusion_matrix(fullfname, confusion_matrix, labels):
    '''
    Save a confusion matrix to a text file.
    
    Confusion matrix comes from evaluations.get_confusion_matrix.
    
    :param str fullfname: The full file name (with path) to the text file.
    :param numpy.ndarray confusion_matrix: The confusion matrix numpy array.
    :param list labels: List of string labels corresponding to the rows/columns of the matrix.
    '''
    with open(fullfname, 'w', encoding='utf-8') as f:
        print(
            '',
            *['true {}'.format(label) for label in labels],
            'total',
            sep='\t', file=f
            )
        for row in range(confusion_matrix.shape[0]):
            print(
                'predicted {}'.format(labels[row]),
                *confusion_matrix[row, :].tolist(),
                confusion_matrix[row, :].sum(),
                sep='\t', file=f
                )
        print(
            'total',
            *[confusion_matrix[:, col].sum() for col in range(confusion_matrix.shape[1])],
            confusion_matrix.sum(),
            sep='\t', file=f
            )


#########################################
class ConfusionMapSaver(object):
    '''
    Confusion map saving helper.
    
    Confusion map comes from evaluations.get_confusion_map.
    '''
    
    #########################################
    def __init__(self):
        '''
        Constructor.
        '''
        labels = [
            (evaluations.TRUE_POSITIVE, 'true +ve'),
            (evaluations.TRUE_NEGATIVE, 'true -ve'),
            (evaluations.FALSE_POSITIVE, 'false +ve'),
            (evaluations.FALSE_NEGATIVE, 'false -ve')
            ]
        labels.sort()
        self.palette = colours.LabelPalette([label for (_, label) in labels])
    
    #########################################
    def save(self, fullfname, confusion_map):
        '''
        Save a confusion map.
        
        :param str fullfname: The full file name (with path) of the image file.
        :param numpy.ndarray confusion_map: The numpy array containing the confusion map.
        '''
        image = self.palette.label_indexes_to_colours(confusion_map)
        images.save_image(fullfname, image, num_bits=8, compress=True)


#########################################
class EvaluationResultsFile(object):
    '''Results file interface for the evaluate command.'''

    #########################################
    def __init__(self, results_fullfname, evaluation):
        '''
        Create an evaluation results file object.

        :param results_fullfname: The full file name (with path) of the results text file. If
            None then no file will be saved and all inputs are ignored.
        :type results_fullfname: str or None
        :param Evaluation evaluation: The evaluation object being used for the results.
        '''
        self.results_fullfname = results_fullfname
        self.total_featuriser_duration = 0.0
        self.total_classifier_duration = 0.0
        self.num_rows = 0
        self.evaluation = evaluation

    #########################################
    def create(self, labels):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'subvolume slice#',
                    'volume slice#',
                    'global {}'.format(self.evaluation.name),
                    'min {}'.format(self.evaluation.name),
                    'stddev {}'.format(self.evaluation.name),
                    *['{} {}'.format(label, self.evaluation.name) for label in labels],
                    'featurisation duration (s)',
                    'prediction duration (s)',
                    sep='\t', file=f
                    )
    
    #########################################
    def load(self):
        '''
        Load the existing results text file to continue adding to it.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'r', encoding='utf-8') as f:
                lines = [line.split('\t') for line in f.read().strip().split('\n')]
            for line in lines[1:]:
                self.total_featuriser_duration += float(line[-2])
                self.total_classifier_duration += float(line[-1])
                self.num_rows += 1

    #########################################
    def add(self, subvolume_slice_num, volume_slice_num, label_results, global_result, featuriser_duration, classifier_duration):
        '''
        Add a new slice's result to the file.

        :param int subvolume_slice_num: The subvolume slice number of the slice being used for
            evaluation.
        :param int volume_slice_num: The full volume slice number of the slice being used for
            evaluation.
        :param list label_results: The list of scores for each label.
        :param float global_result: The global score.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    subvolume_slice_num,
                    volume_slice_num,
                    '{:.3{}}'.format(
                        global_result,
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        min([r for r in label_results if r is not None]),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        np.std([r for r in label_results if r is not None]).tolist(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
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
                    -1,
                    -1,
                    '{:.3{}}'.format(
                        self.evaluation.get_global_result(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        min([r for r in self.evaluation.get_global_result_per_label() if r is not None]),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        np.std([r for r in self.evaluation.get_global_result_per_label() if r is not None]).tolist(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
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
                    '{:.1f}'.format(self.total_featuriser_duration/self.num_rows),
                    '{:.1f}'.format(self.total_classifier_duration/self.num_rows),
                    sep='\t', file=f
                    )


#########################################
class TuningResultsFile(object):
    '''Results file interface for the tune command.'''

    #########################################
    def __init__(self, results_fullfname, evaluation):
        '''
        Create a tune results file object.

        :param results_fullfname: The full file name (with path) of the results text file. If
            None then no file will be saved and all inputs are ignored.
        :type results_fullfname: str or None
        '''
        self.results_fullfname = results_fullfname
        self.evaluation = evaluation
        self.best_config = None
        self.best_globalscore = 0.0

    #########################################
    def create(self, labels, extra_col_names=[]):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        :param list extra_col_names: A list of extra column names to add.
        '''
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'w', encoding='utf-8') as f:
                print(
                    'json config',
                    'global {}'.format(self.evaluation.name),
                    'min {}'.format(self.evaluation.name),
                    'stddev {}'.format(self.evaluation.name),
                    *['{} {}'.format(label, self.evaluation.name) for label in labels],
                    'featuriser duration (s)',
                    'classifier duration (s)',
                    'max memory (MB)',
                    *extra_col_names,
                    sep='\t', file=f
                    )

    #########################################
    def load(self):
        '''Load existing results in order to get the best global evaluation score.'''
        if self.results_fullfname is not None:
            best_jsonconfig = None
            with open(self.results_fullfname, 'r', encoding='utf-8') as f:
                for line in f.read().strip().split('\n')[1:]:
                    [json_config, global_score] = line.split('\t')[:2]
                    if self.evaluation.is_percentage:
                        global_score = global_score[:-1]
                    global_score = float(global_score)
                    if global_score > self.best_globalscore:
                        self.best_globalscore = global_score
                        best_jsonconfig = json_config
            if best_jsonconfig is not None:
                self.best_config = json.loads(best_jsonconfig)
    
    #########################################
    def add(self, config, featuriser_duration, classifier_duration, max_memory_mb, extra_col_values=[]):
        '''
        Add a new result to the file.

        :param dict config: The configuation dictionary used to produce these results.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        :param float max_memory_mb: The maximum number of megabytes of memory used
            during featurisation and classification.
        :param list extra_col_values: A list of extra columns to add.
        '''
        global_score = self.evaluation.get_global_result()
        if global_score > self.best_globalscore:
            self.best_globalscore = global_score
            self.best_config = config
        
        if self.results_fullfname is not None:
            with open(self.results_fullfname, 'a', encoding='utf-8') as f:
                print(
                    json.dumps(config),
                    '{:.3{}}'.format(
                        global_score,
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        min([r for r in self.evaluation.get_global_result_per_label() if r is not None]),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
                    '{:.3{}}'.format(
                        np.std([r for r in self.evaluation.get_global_result_per_label() if r is not None]).tolist(),
                        '%' if self.evaluation.is_percentage else 'f'
                        ),
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
                    '{:.1f}'.format(featuriser_duration),
                    '{:.1f}'.format(classifier_duration),
                    '{:.3f}'.format(max_memory_mb),
                    *extra_col_values,
                    sep='\t', file=f
                    )