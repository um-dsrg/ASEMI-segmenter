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
    def __init__(self, label_names, skip_colours=0):
        '''
        Constructor.
        
        :param list label_names: The names of labels in order of their indexes.
        :param int skip_colours: The number of colours in the sequence to skip.
        '''
        self.label_names = label_names
        self.palette = colours.LabelPalette(label_names, skip_colours)
    
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

        :param results_fullfname: The full file name (with path) of the results text file.
        :type results_fullfname: str or None
        :param Evaluation evaluation: The evaluation object being used for the results.
        '''
        self.results_fullfname = results_fullfname
        self.evaluation = evaluation
        
        self.num_rows = 0
        self.total_featuriser_duration = 0
        self.total_classifier_duration = 0
        self.total_total_duration = 0

    #########################################
    def create(self, labels):
        '''
        Create the results text file.

        :param list labels: The list of labels used in the segmenter.
        '''
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
                'total duration (s)',
                sep='\t', file=f
                )
    
    #########################################
    def load(self, all_predicted_labels, all_true_labels):
        '''
        Load the intermediate results for computing the global scores.
        
        :param list all_predicted_labels: List of numpy array predicted labels
            for every evaluated slice.
        :param list all_true_labels: List of numpy array true labels
            for every evaluated slice.
        '''
        with open(self.results_fullfname, 'r', encoding='utf-8') as f:
            for (i, line) in enumerate(f.read().strip().split('\n')[1:]):
                fields = line.split('\t')
                if int(fields[0]) == -1:
                    continue
                
                self.num_rows += 1
                self.total_featuriser_duration += float(fields[-3])
                self.total_classifier_duration += float(fields[-2])
                self.total_total_duration += float(fields[-1])
        
                self.evaluation.evaluate(all_predicted_labels[i], all_true_labels[i])
    
    #########################################
    def add(self, subvolume_slice_num, volume_slice_num, predicted_labels, true_labels, featuriser_duration, classifier_duration, total_duration):
        '''
        Add a new slice's result to the file.

        :param int subvolume_slice_num: The subvolume slice number of the slice being used for
            evaluation.
        :param int volume_slice_num: The full volume slice number of the slice being used for
            evaluation.
        :param numpy.ndarray predicted_labels: The predicted labels of the slice.
        :param numpy.ndarray true_labels: The true labels of the slice.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        :param float total_duration: The total duration of the slice's evaluation process.
        '''
        self.num_rows += 1
        self.total_featuriser_duration += featuriser_duration
        self.total_classifier_duration += classifier_duration
        self.total_total_duration += total_duration
        
        (label_scores, single_score) = self.evaluation.evaluate(predicted_labels, true_labels)
        with open(self.results_fullfname, 'a', encoding='utf-8') as f:
            print(
                subvolume_slice_num,
                volume_slice_num,
                '{:.3{}}'.format(
                    single_score,
                    '%' if self.evaluation.is_percentage else 'f'
                    ),
                '{:.3{}}'.format(
                    min([r for r in label_scores if r is not None]),
                    '%' if self.evaluation.is_percentage else 'f'
                    ),
                '{:.3{}}'.format(
                    np.std([r for r in label_scores if r is not None]).tolist(),
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
                    for result in label_scores
                    ],
                '{:.1f}'.format(featuriser_duration),
                '{:.1f}'.format(classifier_duration),
                '{:.1f}'.format(total_duration),
                sep='\t', file=f
                )
    
    #########################################
    def conclude(self):
        '''
        Add a global result to the file.
        '''
        (label_scores, single_score) = self.evaluation.get_global_results()
        with open(self.results_fullfname, 'a', encoding='utf-8') as f:
            print(
                -1,
                -1,
                '{:.3{}}'.format(
                    single_score,
                    '%' if self.evaluation.is_percentage else 'f'
                    ),
                '{:.3{}}'.format(
                    min([r for r in label_scores if r is not None]),
                    '%' if self.evaluation.is_percentage else 'f'
                    ),
                '{:.3{}}'.format(
                    np.std([r for r in label_scores if r is not None]).tolist(),
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
                    for result in label_scores
                    ],
                '{:.1f}'.format(self.total_featuriser_duration/self.num_rows),
                '{:.1f}'.format(self.total_classifier_duration/self.num_rows),
                '{:.1f}'.format(self.total_total_duration/self.num_rows),
                sep='\t', file=f
                )


#########################################
class TuningResultsFile(object):
    '''Results file interface for the tune command.'''

    #########################################
    def __init__(self, results_fullfname, evaluation):
        '''
        Create a tune results file object.

        :param results_fullfname: The full file name (with path) of the results text file.
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
        with open(self.results_fullfname, 'w', encoding='utf-8') as f:
            print(
                'json config',
                'global {}'.format(self.evaluation.name),
                'min {}'.format(self.evaluation.name),
                'stddev {}'.format(self.evaluation.name),
                *['{} {}'.format(label, self.evaluation.name) for label in labels],
                'featuriser duration (s)',
                'classifier duration (s)',
                'total duration (s)',
                'max memory (MB)',
                *extra_col_names,
                sep='\t', file=f
                )

    #########################################
    def load(self):
        '''Load existing results in order to get the best global evaluation score.'''
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
    def add(self, config, featuriser_duration, classifier_duration, total_duration, max_memory_mb, extra_col_values=[]):
        '''
        Add a new result to the file.

        :param dict config: The configuation dictionary used to produce these results.
        :param float featuriser_duration: The duration of the featurisation process.
        :param float classifier_duration: The duration of the classification process.
        :param float total_duration: The total duration to compute the row.
        :param float max_memory_mb: The maximum number of megabytes of memory used
            during featurisation and classification.
        :param list extra_col_values: A list of extra columns to add.
        '''
        (label_scores, global_score) = self.evaluation.get_global_results()
        if global_score > self.best_globalscore:
            self.best_globalscore = global_score
            self.best_config = config
        
        with open(self.results_fullfname, 'a', encoding='utf-8') as f:
            print(
                json.dumps(config),
                '{:.3{}}'.format(
                    global_score,
                    '%' if self.evaluation.is_percentage else 'f'
                    ),
                '{:.3{}}'.format(
                    min([r for r in label_scores if r is not None]),
                    '%' if self.evaluation.is_percentage else 'f'
                    ),
                '{:.3{}}'.format(
                    np.std([r for r in label_scores if r is not None]).tolist(),
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
                    for result in label_scores
                    ],
                '{:.1f}'.format(featuriser_duration),
                '{:.1f}'.format(classifier_duration),
                '{:.1f}'.format(total_duration),
                '{:.3f}'.format(max_memory_mb),
                *extra_col_values,
                sep='\t', file=f
                )