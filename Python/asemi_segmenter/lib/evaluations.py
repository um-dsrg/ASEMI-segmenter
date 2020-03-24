'''Evaluation methods for segmenters.'''

import numpy as np
import os
import sys

#########################################
def get_confusion_matrix(predicted_labels, true_labels, num_labels):
    '''
    Create a confusion matrix array.
    
    Labels are numbers between 0 and num_labels-1. Each row stands for a true label and each column
    for a predicted label. Values in the matrix are the number of times a true label was classified
    as a predicted label divided by the total number of true labels that are the given true label
    (total of row).
    
    :param numpy.ndarray predicted_labels: The array of labels given by the segmenter.
    :param numpy.ndarray true_labels: The array of labels given by the dataset.
    :param int num_labels: The number of different labels (or one plus the last label index).
    :return: The confusion matrix.
    :rtype: numpy.ndarray
    '''
    totals = {
            true_label_index: np.sum(true_labels == true_label_index)
            for true_label_index in range(num_labels)
        }
    matrix = np.array([
            [
                np.sum(predicted_labels[true_labels==true_label_index] == predicted_label_index)/totals[true_label_index]
                for predicted_label_index in range(num_labels)
            ]
            for true_label_index in range(num_labels)
        ], np.float32)
    return matrix
    
#########################################
def get_classification_accuracies(predicted_labels, true_labels, num_labels):
    '''
    The per label accuracy of the segmentation.
    
    For a given label, the accuracy is the number of correctly predicted labels divided by the
    number of times that label is in the dataset.
    
    :param numpy.ndarray predicted_labels: The array of labels given by the segmenter.
    :param numpy.ndarray true_labels: The array of labels given by the dataset.
    :param int num_labels: The number of different labels (or one plus the last label index).
    :return: The list of accuracies (one for each label).
    :rtype: list
    '''
    accuracies = [
            (np.sum(true_labels[true_labels==label_index] == predicted_labels[true_labels==label_index])/np.sum(true_labels==label_index)).tolist()
            for label_index in range(num_labels)
        ]
    return accuracies

#########################################
def get_intersection_over_union(predicted_labels, true_labels, num_labels):
    '''
    The per label IoU (intersection over union) of the segmentation.
    
    For a given label, the IoU is the number of correctly predicted labels divided by the total of:
    * the number of correct predctions
    * the number of labels that were supposed to be classified as the given label but weren't
    * the number of labels that were classified as the given label but weren't suppose to be
    
    :param numpy.ndarray predicted_labels: The array of labels given by the segmenter.
    :param numpy.ndarray true_labels: The array of labels given by the dataset.
    :param int num_labels: The number of different labels (or one plus the last label index).
    :return: The list of IoUs (one for each label).
    :rtype: list
    '''
    if len(predicted_labels.shape) == 2:
        predicted_labels = np.reshape(predicted_labels, (1,) + predicted_labels.shape)
    if len(true_labels.shape) == 2:
        true_labels = np.reshape(true_labels, (1,) + true_labels.shape)
    accuracies = list()
    for label_index in range(num_labels):
        intersection = 0
        union = 0
        num_trues = 0
        for i in range(predicted_labels.shape[0]):
            trues = true_labels == label_index
            preds = predicted_labels == label_index
            num_trues += np.sum(trues)
            intersection += np.sum(trues*preds)
            union += np.sum(trues + preds)
        if num_trues == 0:
            accuracies.append(None)
        else:
            accuracies.append(intersection/union)
    return accuracies