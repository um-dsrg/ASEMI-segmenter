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

import argparse
import numpy as np
import memory_profiler
import tempfile
import shutil
import json
import os
import sys
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import times
from asemi_segmenter.lib import files
from asemi_segmenter.lib import images
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import checkpoints
from asemi_segmenter import preprocess
from asemi_segmenter import train
from asemi_segmenter import segment

#########################################
def measure(results_fullfname, side_lengths, max_batch_memory_list, max_processes_featuriser_list, max_processes_classifier_list, use_gpu, num_training_slices, num_labels, num_runs, train_config, segment_config, num_simultaneous_slices, voxel_seed, label_seed):
    checkpoint = checkpoints.CheckpointManager(
        'timespace',
        'checkpoint.json'
        )

    with checkpoint.apply('create_file') as skip:
        if skip is not None:
            print('Reusing results file')
            print()
            raise skip
        with open(results_fullfname, 'w', encoding='utf-8') as f:
            print('side_length', 'max_batch_memory', 'max_processes_featuriser', 'max_processes_classifier', 'slice_area', 'cube_volume', 'run', 'train_time', 'train_memory', 'segment_time', 'segment_memory', sep='\t', file=f)

    featuriser = featurisers.load_featuriser_from_config(train_config['featuriser'])
    preprocess_config = {
            'num_downsamples': max(featuriser.get_scales_needed()),
            'downsample_filter': {
                    'type': 'none',
                    'params': {
                        }
                },
            'hash_function': {
                    'type': 'random_indexing',
                    'params': {
                            'hash_size': 10
                        }
                }
        }
    del featuriser

    for side_length in side_lengths:
        print(times.get_timestamp())
        print('Side length:', side_length)
        print()

        with checkpoint.apply('{}'.format(side_length)) as skip:
            if skip is not None:
                print(' Skipped as was found checkpointed')
                raise skip

            with tempfile.TemporaryDirectory(dir='.') as temp_dir:
                print(' Creating full volume slices')
                files.mkdir(os.path.join(temp_dir, 'volume'))
                r = np.random.RandomState(voxel_seed)
                for i in range(side_length):
                    images.save_image(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), r.randint(0, 2**16, size=[side_length,side_length], dtype=np.uint16))

                print(' Creating sub volume slices')
                files.mkdir(os.path.join(temp_dir, 'subvolume'))
                for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
                    shutil.copy(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), os.path.join(temp_dir, 'subvolume', '{}.tif'.format(i)))

                print(' Creating label slices')
                files.mkdir(os.path.join(temp_dir, 'labels'))
                r = np.random.RandomState(label_seed)
                for label in range(num_labels):
                    files.mkdir(os.path.join(temp_dir, 'labels', '_{}'.format(label)))
                    for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
                        images.save_image(os.path.join(temp_dir, 'labels', '_{}'.format(label), '{}.tif'.format(i)), r.randint(0, 2, size=[side_length,side_length], dtype=np.uint16))

                print(' Preprocessing volume')
                preprocess.main(
                        volume_dir=os.path.join(temp_dir, 'volume'),
                        config=preprocess_config,
                        result_data_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                        checkpoint_fullfname=None,
                        reset_checkpoint=False,
                        checkpoint_namespace=None,
                        checkpoint_init=None,
                        max_processes=max(max(max_processes_featuriser_list_list), max(max_processes_classifier_list)),
                        max_batch_memory=max(max_batch_memory_list),
                        debug_mode=True
                    )

                files.mkdir(os.path.join(temp_dir, 'segmentation'))

                print(' Starting measurements')
                for max_batch_memory in max_batch_memory_list:
                    print('  Side length:', side_length)
                    print('  Max batch memory:', max_batch_memory)
                    for max_processes_featuriser in max_processes_featuriser_list:
                        print('   Max processes featuriser:', max_processes_featuriser)
                        for max_processes_classifier in max_processes_classifier_list:
                            print('   Max processes classifier:', max_processes_classifier)
                            for run in range(1, num_runs+1):
                                with checkpoint.apply('{}-{}-{}-{}-{}'.format(side_length, max_batch_memory, max_processes_featuriser, max_processes_classifier, run)) as skip:
                                    if skip is not None:
                                        print('    Measuring run {} (Skipped as was found checkpointed)'.format(run))
                                        raise skip
                                    else:
                                        print('    Measuring run {} ({})'.format(run, times.get_timestamp()))

                                    with times.Timer() as timer:
                                        mem_usage = memory_profiler.memory_usage(
                                            (
                                                train.main,
                                                [],
                                                dict(
                                                    preproc_volume_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                                                    subvolume_dir=os.path.join(temp_dir, 'subvolume'),
                                                    label_dirs=[ os.path.join(temp_dir, 'labels', '_{}'.format(label)) for label in range(num_labels) ],
                                                    config=train_config,
                                                    result_segmenter_fullfname=os.path.join(temp_dir, 'segmenter.pkl'),
                                                    trainingset_file_fullfname=None,
                                                    checkpoint_fullfname=None,
                                                    reset_checkpoint=False,
                                                    checkpoint_namespace=None,
                                                    checkpoint_init=None,
                                                    max_processes_featuriser=max_processes_featuriser,
                                                    max_processes_classifier=max_processes_classifier,
                                                    max_batch_memory=max_batch_memory,
                                                    use_gpu=use_gpu,
                                                    debug_mode=True
                                                )
                                            )
                                        )

                                    train_time = timer.duration
                                    train_memory = max(mem_usage)

                                    with times.Timer() as timer:
                                        mem_usage = memory_profiler.memory_usage(
                                            (
                                                segment.main,
                                                [],
                                                dict(
                                                    segmenter=os.path.join(temp_dir, 'segmenter.pkl'),
                                                    preproc_volume_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                                                    config=segment_config,
                                                    results_dir=os.path.join(temp_dir, 'segmentation'),
                                                    checkpoint_fullfname=None,
                                                    reset_checkpoint=False,
                                                    checkpoint_namespace=None,
                                                    checkpoint_init=None,
                                                    num_simultaneous_slices=num_simultaneous_slices,
                                                    max_processes_featuriser=max_processes_featuriser,
                                                    max_processes_classifier=max_processes_classifier,
                                                    max_batch_memory=max_batch_memory,
                                                    use_gpu=use_gpu,
                                                    debug_mode=True
                                                )
                                            )
                                        )
                                    segment_time = timer.duration
                                    segment_memory = max(mem_usage)

                                    with open(results_fullfname, 'a', encoding='utf-8') as f:
                                        print(side_length, max_batch_memory, max_processes_featuriser, max_processes_classifier, side_length**2, side_length**3, run, round(train_time, 1), train_memory, round(segment_time, 1), segment_memory, sep='\t', file=f)

                        print()

    print(times.get_timestamp())


#########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_fullfname', required=True,
        help='Full file name (with path) of the train config file.')
    parser.add_argument('--segment_config_fullfname', required=True,
        help='Full file name (with path) of the segment config file.')
    parser.add_argument('--results_fullfname', required=True,
        help='Full file name (with path) of the results text file.')
    parser.add_argument('--max_processes_featuriser', nargs='+', required=True, type=int,
        help='The different maximum number of processes to use whilst featurising.')
    parser.add_argument('--max_processes_classifier', nargs='+', required=True, type=int,
        help='The different maximum number of processes to use whilst classifying.')
    parser.add_argument('--max_batch_memory', nargs='+', required=True, type=float,
        help='The different maximum amount of memory in GB to use.')
    parser.add_argument('--side_length', nargs='+', required=True, type=int,
        help='The different lengths of the cube volumes to process.')
    parser.add_argument('--num_training_slices', required=True, type=int,
        help='The number of training slices to take from the volume.')
    parser.add_argument('--num_labels', required=True, type=int,
        help='The number of different labels to put in the training slices.')
    parser.add_argument('--num_runs', required=True, type=int,
        help='The number of times to run each measurement.')
    parser.add_argument('--num_simultaneous_slices', required=True, type=int,
        help='The number of adjacent slices to process together.')
    parser.add_argument('--voxel_seed', required=False, default=None, type=int,
        help='Seed for the random number generator to generate voxel values. If left out then it will be non-deterministic.')
    parser.add_argument('--label_seed', required=False, default=None, type=int,
        help='Seed for the random number generator to generate labels. If left out then it will be non-deterministic.')
    parser.add_argument('--use_gpu', required=False, default='no', choices=['yes', 'no'],
        help='Whether to use the GPU for computing feature.')
    args = parser.parse_args()

    with open(args.train_config_fullfname, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    with open(args.segment_config_fullfname, 'r', encoding='utf-8') as f:
        segment_config = json.load(f)

    measure(args.results_fullfname, args.side_length, args.max_batch_memory, args.max_processes_featuriser, args.max_processes_classifier, args.use_gpu == 'yes', args.num_training_slices, args.num_labels, args.num_runs, train_config, segment_config, args.num_simultaneous_slices, args.voxel_seed, args.label_seed)


# main entry point
if __name__ == '__main__':
   main()

print('Ready.')
