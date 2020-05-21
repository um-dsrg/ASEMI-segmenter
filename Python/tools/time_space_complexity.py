#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

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
from asemi_segmenter import preprocess
from asemi_segmenter import train
from asemi_segmenter import segment
from asemi_segmenter import listener

class Listener(listener.ProgressListener):
    def error_output(self, text):
        raise Exception(text)

#########################################
def measure(side_length, num_training_slices, num_labels, num_runs, train_config, max_processes, max_batch_memory, listener=lambda run, train_time, train_memory, segment_time, segment_memory:None):
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

    with tempfile.TemporaryDirectory() as temp_dir:
        print('Creating full volume slices')
        files.mkdir(os.path.join(temp_dir, 'volume'))
        r = np.random.RandomState(0)
        for i in range(side_length):
            images.save_image(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), r.randint(0, 2**16, size=[side_length,side_length], dtype=np.uint16))

        print('Creating sub volume slices')
        files.mkdir(os.path.join(temp_dir, 'subvolume'))
        for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
            shutil.copy(os.path.join(temp_dir, 'volume', '{}.tif'.format(i)), os.path.join(temp_dir, 'subvolume', '{}.tif'.format(i)))

        print('Creating label slices')
        files.mkdir(os.path.join(temp_dir, 'labels'))
        r = np.random.RandomState(0)
        for label in range(num_labels):
            files.mkdir(os.path.join(temp_dir, 'labels', '_{}'.format(label)))
            for i in range(0, side_length-side_length%num_training_slices, side_length//num_training_slices):
                images.save_image(os.path.join(temp_dir, 'labels', '_{}'.format(label), '{}.tif'.format(i)), r.randint(0, 2, size=[side_length,side_length], dtype=np.uint16))

        print('Preprocessing volume')
        preprocess.main(
                volume_dir=os.path.join(temp_dir, 'volume'),
                config=preprocess_config,
                result_data_fullfname=os.path.join(temp_dir, 'full_volume.hdf'),
                checkpoint_fullfname=None,
                checkpoint_namespace='',
                reset_checkpoint=False,
                checkpoint_init=None,
                max_processes=max_processes,
                max_batch_memory=max_batch_memory,
                listener=Listener()
            )

        files.mkdir(os.path.join(temp_dir, 'segmentation'))

        print('Starting measurements')
        for run in range(1, num_runs+1):
            print('Measuring run', run)
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
                            checkpoint_namespace='',
                            reset_checkpoint=False,
                            checkpoint_init=None,
                            max_processes=max_processes,
                            max_batch_memory=max_batch_memory,
                            listener=Listener()
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
                            config={
                                    "soft_segmentation": False,
                                    "as_masks": True,
                                    "image_extension": "tiff",
                                    "bits": 8
                                },
                            results_dir=os.path.join(temp_dir, 'segmentation'),
                            checkpoint_fullfname=None,
                            checkpoint_namespace='',
                            reset_checkpoint=False,
                            checkpoint_init=None,
                            max_processes=max_processes,
                            max_batch_memory=max_batch_memory,
                            listener=Listener()
                        )
                    )
                )
            segment_time = timer.duration
            segment_memory = max(mem_usage)

            listener(run, train_time, train_memory, segment_time, segment_memory)

        print()


#########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config_fullfname', required=True,
        help='Full file name (with path) of the train config file.')
    parser.add_argument('--results_fullfname', required=True,
        help='Full file name (with path) of the results text file.')
    parser.add_argument('--max_processes', nargs='+', required=True, type=int,
        help='The different maximum number of processes to use.')
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
    args = parser.parse_args()
    
    print('Running...')
    print()

    with open(args.train_config_fullfname, 'r', encoding='utf-8') as f:
        train_config = json.load(f)

    with open(args.results_fullfname, 'w', encoding='utf-8') as f:
        print('max_batch_memory', 'max_processes', 'side_length', 'slice_area', 'cube_volume', 'run', 'train_time', 'train_memory', 'segment_time', 'segment_memory', sep='\t', file=f)

    for max_batch_memory in args.max_batch_memory:
        for max_processes in args.max_processes:
            for side_length in args.side_length:
                print('max_batch_memory', max_batch_memory, '-', 'max_processes', max_processes, '-', 'side_length', side_length)
                def listener(run, train_time, train_memory, segment_time, segment_memory):
                    with open(args.results_fullfname, 'a', encoding='utf-8') as f:
                        print(max_batch_memory, max_processes, side_length, side_length**2, side_length**3, run, train_time, train_memory, segment_time, segment_memory, sep='\t', file=f)

                measure(side_length, args.num_training_slices, args.num_labels, args.num_runs, train_config, max_processes, max_batch_memory, listener)


# main entry point
if __name__ == '__main__':
   main()
