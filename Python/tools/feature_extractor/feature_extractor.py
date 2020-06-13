#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

import os
import argparse
import json
import numpy as np
import memory_profiler
from asemi_segmenter.lib import times
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import arrayprocs


#########################################
def extract_features(preproc_volume_fullfname, featuriser_config, volume_slice_index, volume_slice_count, max_processes, max_batch_memory, save_as=None, use_gpu=False):
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load()

    featuriser = featurisers.load_featuriser_from_config(featuriser_config, use_gpu=use_gpu)

    best_block_shape = arrayprocs.get_optimal_block_size(
        full_volume.get_shape()[1:],
        full_volume.get_dtype(),
        featuriser.get_context_needed(),
        max_processes,
        max_batch_memory,
        num_implicit_slices=volume_slice_count,
        feature_size=featuriser.get_feature_size(),
        feature_dtype=featurisers.feature_dtype
        )

    result = []
    def f(result):
        result.append(
            featuriser.featurise_slice(
                full_volume.get_scale_arrays(featuriser.get_scales_needed()),
                slice_range=slice(volume_slice_index, volume_slice_index+volume_slice_count),
                block_shape=best_block_shape,
                n_jobs=max_processes
                )
            )

    with times.Timer() as timer:
        mem_usage = memory_profiler.memory_usage((f, (result,)))
    feature_vectors = result[0]

    print('duration:', round(timer.duration, 1), 's')
    print('memory:', round(max(mem_usage), 2), 'MB')

    if save_as is not None:
        if save_as.endswith('.txt'):
            with open(save_as, 'w', encoding='utf-8') as f:
                for feature_vector in feature_vectors.tolist():
                    print(*feature_vector, sep='\t', file=f)
        if save_as.endswith('.npy'):
            with open(save_as, 'wb') as f:
                np.save(f, feature_vectors, allow_pickle=False)


#########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_fullfname', required=True,
        help='Full file name (with path) of volume file featurise.')
    parser.add_argument('--feature_config_fullfname', required=True,
        help='JSON file containing the featuriser configuration.')
    parser.add_argument('--volume_slice_index', required=True, type=int,
        help='The first slice index to featurise in the volume.')
    parser.add_argument('--volume_slice_count', required=True, type=int,
        help='The number of slices to featurise in the volume.')
    parser.add_argument('--save_as', required=False, default=None,
        help='Full file name (with path) of file to contain the features '
            '(can be .txt, .npy, or left out to not save anything).')
    parser.add_argument('--max_processes', required=True, type=int,
        help='The maximum number of processes to use.')
    parser.add_argument('--max_batch_memory', required=True, type=float,
        help='The maximum amount of memory in GB to use.')
    parser.add_argument('--use_gpu', required=False, default='no', choices=['yes', 'no'],
        help='Whether to use the GPU for computing feature.')
    args = parser.parse_args()

    with open(args.feature_config_fullfname, 'r', encoding='utf-8') as f:
        featuriser_config = json.load(f)

    print('Running...')

    extract_features(
        args.volume_fullfname,
        featuriser_config,
        volume_slice_index=args.volume_slice_index,
        volume_slice_count=args.volume_slice_count,
        save_as=args.save_as,
        max_processes=2,
        max_batch_memory=0.1,
        use_gpu=args.use_gpu == 'yes'
        )


# main entry point
if __name__ == '__main__':
   main()
