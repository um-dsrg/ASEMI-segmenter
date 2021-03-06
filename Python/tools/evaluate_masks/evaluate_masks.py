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
from asemi_segmenter.lib import colours

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groundtruth_labels', required=True, nargs='+',
        help='Folders with correct segmentation label masks for each label.')
    parser.add_argument('--predicted_labels', required=True, nargs='+',
        help='Folders with predicted segmentation label masks for each label.')
    parser.add_argument('--results_dir', required=True,
        help='The path to the directory to contain the results of this command.')
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

    assert groundtruth_labels == predicted_labels, (groundtruth_labels, predicted_labels)
    assert len(groundtruth_labels_data[0].fullfnames) == len(predicted_labels_data[0].fullfnames), (len(groundtruth_labels_data[0].fullfnames), len(predicted_labels_data[0].fullfnames))

    labels = groundtruth_labels
    num_slices = len(groundtruth_labels_data[0].fullfnames)
    slice_shape = groundtruth_labels_data[0].shape
    slice_size = slice_shape[0]*slice_shape[1]
    num_digits_in_filename = math.ceil(math.log10(len(num_slices)+1))

    print('Loading groundtruth labels')
    all_groundtruths = volumes.load_labels(groundtruth_labels_data)
    print('Loading predicted labels')
    all_predictions = volumes.load_labels(predicted_labels_data)

    evaluation = evaluations.IntersectionOverUnionEvaluation(len(labels))
    evaluation_results_file = results.EvaluationResultsFile(os.path.join(args.results_dir, 'results.txt'), evaluation)
    evaluation_results_file.create(labels)

    labels_palette = colours.LabelPalette([''] + labels)
    images.save_image(
        os.path.join(args.results_dir, 'legend.tiff'),
        images.matplotlib_to_imagedata(labels_palette.get_legend()),
        num_bits=8,
        compress=True
        )
    confusion_map_saver = results.ConfusionMapSaver(labels, skip_colours=1)

    for i in range(num_slices):
        print('Evaluating slice #{}'.format(i + 1))
        groundtruth_slice_labels = all_groundtruths[i*slice_size:(i+1)*slice_size]
        predicted_slice_labels = all_predictions[i*slice_size:(i+1)*slice_size]

        files.mkdir(os.path.join(args.results_dir, 'slice_{:0>{}d}'.format(i + 1, num_digits_in_filename)))

        reshaped_groundtruth_slice_labels = groundtruth_slice_labels.reshape(slice_shape)
        reshaped_prediction_slice_labels = predicted_slice_labels.reshape(slice_shape)

        images.save_image(
            os.path.join(args.results_dir, 'slice_{:0>{}d}'.format(i + 1, num_digits_in_filename), 'true_labels.tiff'),
            labels_palette.label_indexes_to_colours(
                np.where(
                    reshaped_groundtruth_slice_labels >= volumes.FIRST_CONTROL_LABEL,
                    0,
                    reshaped_groundtruth_slice_labels + 1
                    )
                ),
            num_bits=8,
            compress=True
            )

        images.save_image(
            os.path.join(args.results_dir, 'slice_{:0>{}d}'.format(i + 1, num_digits_in_filename), 'predicted_labels.tiff'),
            labels_palette.label_indexes_to_colours(
                np.where(
                    reshaped_prediction_slice_labels >= volumes.FIRST_CONTROL_LABEL,
                    0,
                    reshaped_prediction_slice_labels + 1
                    )
                ),
            num_bits=8,
            compress=True
            )

        confusion_matrix = evaluations.get_confusion_matrix(
            reshaped_prediction_slice_labels,
            reshaped_groundtruth_slice_labels,
            len(labels)
            )
        results.save_confusion_matrix(
            os.path.join(args.results_dir, 'slice_{:0>{}d}'.format(i + 1, num_digits_in_filename), 'confusion_matrix.txt'),
            confusion_matrix,
            labels
            )

        for (label_index, label) in enumerate(labels):
            confusion_map = evaluations.get_confusion_map(
                reshaped_prediction_slice_labels,
                reshaped_groundtruth_slice_labels,
                label_index
                )
            confusion_map_saver.save(
                os.path.join(args.results_dir, 'slice_{:0>{}d}'.format(i + 1, num_digits_in_filename), 'confusion_map_{}.tiff'.format(label)),
                confusion_map,
                label_index
                )

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

    global_confusion_matrix = np.zeros(
        (len(labels), len(labels)),
        np.uint64
        )
    for i in range(num_slices):
        with open(os.path.join(args.results_dir, 'slice_{:0>{}d}'.format(i + 1, num_digits_in_filename), 'confusion_matrix.txt'), 'r', encoding='utf-8') as f:
            confusion_matrix = np.array([
                [ int(str_freq) for str_freq in line.split('\t')[1:-1] ]
                for line in f.read().strip().split('\n')[1:-1]
                ],
                np.uint64
                )
        global_confusion_matrix += confusion_matrix

    results.save_confusion_matrix(
        os.path.join(args.results_dir, 'global_confusion_matrix.txt'),
        global_confusion_matrix,
        labels
        )

if __name__ == '__main__':
   main()
