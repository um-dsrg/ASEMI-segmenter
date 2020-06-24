#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

import os
import math
import PIL.Image
import numpy as np
import argparse
from asemi_segmenter.lib import images
from asemi_segmenter.lib import files
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import results
from asemi_segmenter.lib import evaluations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth_labels', required=True, nargs='+',
        help='Folders with correct segmentation label masks for each label.')
    parser.add_argument('--predicted_labels', required=True, nargs='+',
        help='Folders with predicted segmentation label masks for each label.')
    parser.add_argument('--results_fullfname', required=True,
        help='Full file name (with path) to output results text file to create (.txt).')
    args = parser.parse_args()

    print('Running...')

    groundtruth_labels_data = []
    for label_dir in args.groundtruth_labels:
        label_data = volumes.load_label_dir(label_dir)
        groundtruth_labels_data.append(label_data)
    groundtruth_labels = sorted(label_data.name for label_data in groundtruth_labels_data)

    predicted_labels_data = []
    for label_dir in args.predicted_labels:
        label_data = volumes.load_label_dir(label_dir)
        predicted_labels_data.append(label_data)
    predicted_labels = sorted(label_data.name for label_data in predicted_labels_data)

    assert groundtruth_labels == predicted_labels
    assert len(groundtruth_labels_data[0].fullfnames) == len(predicted_labels_data[0].fullfnames)

    labels = groundtruth_labels
    num_slices = len(groundtruth_labels_data[0].fullfnames)
    slice_shape = groundtruth_labels_data[0].shape
    slice_size = slice_shape[0]*slice_shape[1]

    evaluation = evaluations.IntersectionOverUnionEvaluation(len(labels))
    evaluation_results_file = results.EvaluationResultsFile(args.results_fullfname, evaluation)
    evaluation_results_file.create(labels)

    print('Loading groundtruth labels')
    all_groundtruths = volumes.load_labels(groundtruth_labels_data)
    print('Loading predicted labels')
    all_predictions = volumes.load_labels(predicted_labels_data)

    for i in range(num_slices):
        print('Evaluating slice #{}'.format(i + 1))
        groundtruth_slice_labels = all_groundtruths[i*slice_size:(i+1)*slice_size]
        predicted_slice_labels = all_predictions[i*slice_size:(i+1)*slice_size]

        evaluation_results_file.add(
            i + 1,
            -1,
            predicted_slice_labels,
            groundtruth_slice_labels,
            -1,
            -1,
            -1
            )

    evaluation_results_file.conclude()

if __name__ == '__main__':
   main()
