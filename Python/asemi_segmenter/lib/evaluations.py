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

'''Evaluation methods for segmenters.'''

import numpy as np
import os
import sys

#########################################
def get_confusion_matrix(predicted_labels, true_labels, num_labels):
    '''
    Create a confusion matrix array.

    Labels are numbers between 0 and num_labels-1. Each row stands for a true label and each
    column for a predicted label. Values in the matrix are the number of times a true label
    was classified as a predicted label.

    :param numpy.ndarray predicted_labels: The array of labels given by the segmenter.
    :param numpy.ndarray true_labels: The array of labels given by the dataset.
    :param int num_labels: The number of different labels (or one plus the last label index).
    :return: The confusion matrix.
    :rtype: numpy.ndarray
    '''
    matrix = np.empty((num_labels, num_labels), np.uint64)
    for predicted_index in range(num_labels):
        predicted_label_mask = predicted_labels == predicted_index
        for true_index in range(num_labels):
            true_label_mask = true_labels == true_index
            matrix[predicted_index, true_index] = np.sum(
                np.logical_and(predicted_label_mask, true_label_mask)
                )
    return matrix


#########################################
def get_confusion_map(predicted_labels, true_labels, label_index):
    '''
    Create a confusion map array for a given label.

    A confusion map takes an array of predicted labels and of true labels and,
    for a given reference label, replaces all correct predictions with the
    reference label whilst preserving the incorrect prediction labels. This
    let's one analyse where the errors were made.

    :param numpy.ndarray predicted_labels: The array of labels given by the segmenter.
    :param numpy.ndarray true_labels: The array of labels given by the dataset.
    :param int label_index: The label index to consider.
    :return: The confusion map.
    :rtype: numpy.ndarray
    '''
    predicted_label_mask = predicted_labels == label_index
    true_label_mask = true_labels == label_index
    correct_predictions = (
        (predicted_labels == label_index)
        ==
        (true_labels == label_index)
        )

    return np.where(correct_predictions, label_index, predicted_labels)


#########################################
def load_evaluation_from_config(config, num_labels):
    '''
    Load an evaluation method from a configuration dictionary.

    :param dict config: Configuration of the evaluation method.
    :return: An evaluation object.
    :rtype: Evaluation
    '''
    validations.validate_json_with_schema_file(config, 'evaluation.json')

    if config['type'] == 'accuracy':
        return AccuracyEvaluation(num_labels)
    elif config['type'] == 'iou':
        return IntersectionOverUnionEvaluation(num_labels)


#########################################
class Evaluation(object):
    '''
    Base class for all evaluation methods.

    This object is meant to be used on several slices via separate calls to the
    evaluate method. The object will keep track of some metric for each slice
    and then calculate single weighted mean metric for each label or globally.
    Frequency of each label is also recorded. The reset method can be used to
    clear the internal memory.
    '''

    #########################################
    def __init__(self, num_labels, is_percentage, name):
        '''
        Constructor.

        :param int num_labels: The number of different labels.
        :param bool is_percentage: Whether the .
        :param str name: The name of the evaluation method.
        '''
        self.name = name
        self.is_percentage = is_percentage
        self.num_labels = num_labels
        self.label_freqs = [0 for _ in range(num_labels)]

    #########################################
    def reset(self):
        '''
        Clear recorded slice metrics of object.
        '''
        raise NotImplementedError()

    #########################################
    def get_global_results(self):
        '''
        Convert all separate evaluations into a single number for each label globally.

        Missing labels in the ground truth will be None in label result and ignored
        in global result. If all labels are missing then global result will be None.

        :return: A tuple consisting of a list of global scores (floats) for each label
            and a float being the final global result.
        :rtype: tuple
        '''
        raise NotImplementedError()

    #########################################
    def evaluate(self, predicted_labels, true_labels):
        '''
        Get an evaluation metric number for each label in the input.

        Missing labels in the ground truth will be given None. Results are
        recorded until reset method is called.

        :param numpy.ndarray predicted_labels: A 1D numpy array of label
            indexes given by the segmenter.
        :param numpy.ndarray true_labels: A 1D numpy array of label
            indexes given by the dataset.
        :return: A tuple consisting of a list of scores (floats) for each label
            and a float being global result for this evaluation.
        :rtype: tuple
        '''
        raise NotImplementedError()


#########################################
class AccuracyEvaluation(Evaluation):
    '''
    The per label accuracy of the segmentation.

    For a given label, the accuracy is the number of correctly predicted labels divided by the
    number of times that label is in the dataset.
    '''

    #########################################
    def __init__(self, num_labels, name='acc'):
        '''
        Constructor.

        :param int num_labels: The number of different labels.
        :param str name: The name of the evaluation method.
        '''
        super().__init__(num_labels, True, name)
        self.num_correct = [0 for _ in range(num_labels)]

    #########################################
    def reset(self):
        '''
        Clear recorded slice metrics of object.
        '''
        for label_index in range(num_labels):
            self.label_freqs[label_index] = 0
            self.num_correct[label_index] = 0

    #########################################
    def get_global_results(self):
        '''
        Convert all separate evaluations into a single number for each label globally.

        Missing labels in the ground truth will be None in label result and ignored
        in global result. If all labels are missing then global result will be None.

        :return: A tuple consisting of a list of global scores (floats) for each label
            and a float being the final global result.
        :rtype: tuple
        '''
        total_num_correct = sum(self.num_correct)
        total_label_freqs = sum(self.label_freqs)
        return (
            [
                self.num_correct[label_index]/self.label_freqs[label_index]
                if self.label_freqs[label_index] > 0
                else None
                for label_index in range(self.num_labels)
                ],
            (
                total_num_correct/total_label_freqs
                if total_label_freqs > 0
                else None
                )
            )

    #########################################
    def evaluate(self, predicted_labels, true_labels):
        '''
        Get an evaluation metric number for each label in the input.

        Missing labels in the ground truth will be given None. Results are
        recorded until reset method is called.

        :param numpy.ndarray predicted_labels: A 1D numpy array of label
            indexes given by the segmenter.
        :param numpy.ndarray true_labels: A 1D numpy array of label
            indexes given by the dataset.
        :return: A tuple consisting of a list of scores (floats) for each label
            and a float being global result for this evaluation.
        :rtype: tuple
        '''
        label_freqs = list()
        num_correct = list()
        labels_result = list()
        for label_index in range(self.num_labels):
            label_mask = true_labels == label_index

            label_freq = np.sum(label_mask).tolist()
            label_correct = np.sum(
                true_labels[label_mask] == predicted_labels[label_mask]
                ).tolist()

            label_freqs.append(label_freq)
            num_correct.append(label_correct)
            labels_result.append(label_correct/label_freq if label_correct > 0 else None)

            self.label_freqs[label_index] += label_freqs[label_index]
            self.num_correct[label_index] += num_correct[label_index]
        global_result = sum(num_correct)/sum(label_freqs) if label_correct > 0 else None
        return (labels_result, global_result)


#########################################
class IntersectionOverUnionEvaluation(Evaluation):
    '''
    The per label IoU (intersection over union) of the segmentation.

    For a given label, the IoU is the number of correctly predicted labels divided by the total of:
    * the number of correct predictions
    * the number of labels that were supposed to be classified as the given label but weren't
    * the number of labels that were classified as the given label but weren't suppose to be
    '''

    #########################################
    def __init__(self, num_labels, name='iou'):
        '''
        Constructor.

        :param int num_labels: The number of different labels.
        :param str name: The name of the evaluation method.
        '''
        super().__init__(num_labels, True, name)
        self.num_intersecting = [0 for _ in range(num_labels)]
        self.num_unioned = [0 for _ in range(num_labels)]

    #########################################
    def reset(self):
        '''
        Clear recorded slice metrics of object.
        '''
        for label_index in range(self.num_labels):
            self.label_freqs[label_index] = 0
            self.num_intersecting[label_index] = 0
            self.num_unioned[label_index] = 0

    #########################################
    def get_global_results(self):
        '''
        Convert all separate evaluations into a single number for each label globally.

        Missing labels in the ground truth will be None in label result and ignored
        in global result. If all labels are missing then global result will be None.

        :return: A tuple consisting of a list of global scores (floats) for each label
            and a float being the final global result.
        :rtype: tuple
        '''
        total_num_intersecting = sum(self.num_intersecting)
        total_valid = sum(self.label_freqs)
        return (
            [
                self.num_intersecting[label_index]/self.num_unioned[label_index]
                if self.num_unioned[label_index] > 0
                else None
                for label_index in range(self.num_labels)
                ],
            (
                total_num_intersecting/total_valid
                if total_valid > 0
                else None
                )
            )

    #########################################
    def evaluate(self, predicted_labels, true_labels):
        '''
        Get an evaluation metric number for each label in the input.

        Missing labels in the ground truth will be given None. Results are
        recorded until reset method is called.

        :param numpy.ndarray predicted_labels: A 1D numpy array of label
            indexes given by the segmenter.
        :param numpy.ndarray true_labels: A 1D numpy array of label
            indexes given by the dataset.
        :return: A tuple consisting of a list of scores (floats) for each label
            and a float being the global result for this evaluation.
        :rtype: tuple
        '''
        label_freqs = list()
        num_intersecting = list()
        num_unioned = list()
        labels_result = list()
        for label_index in range(self.num_labels):
            label_mask = true_labels == label_index
            valid_mask = true_labels < self.num_labels
            pred_mask = (predicted_labels == label_index)*valid_mask

            label_freq = np.sum(label_mask).tolist()
            label_intersection = np.sum(label_mask*pred_mask).tolist()
            label_union = np.sum(label_mask + pred_mask).tolist()

            label_freqs.append(label_freq)
            num_intersecting.append(label_intersection)
            num_unioned.append(label_union)
            labels_result.append(label_intersection/label_union if label_freq > 0 else None)

            self.label_freqs[label_index] += label_freqs[label_index]
            self.num_intersecting[label_index] += num_intersecting[label_index]
            self.num_unioned[label_index] += num_unioned[label_index]
        single_result = sum(num_intersecting)/sum(label_freqs) if any(f > 0 for f in label_freqs) else None
        return (labels_result, single_result)
