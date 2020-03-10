import numpy as np
import os
import sys
sys.path.append(os.path.join('..', 'lib'))

#########################################
def get_confusion_matrix(predicted_labels, true_labels, num_labels):
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
    accuracies = [
            (np.sum(true_labels[true_labels==label_index] == predicted_labels[true_labels==label_index])/np.sum(true_labels==label_index)).tolist()
            for label_index in range(num_labels)
        ]
    return accuracies

#########################################
def get_intersection_over_union(predicted_labels, true_labels, num_labels):
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