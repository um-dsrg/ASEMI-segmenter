#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

import argparse
import cProfile
import pstats
import os
import io
from asemi_segmenter import listeners
from asemi_segmenter import segment


#########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_fullfname', required=True,
        help='Full file name (with path) of volume file segment.')
    parser.add_argument('--segmenter_fullfname', required=True,
        help='Full file name (with path) of the saved segmenter.')
    parser.add_argument('--segment_config_fullfname', required=True,
        help='Full file name (with path) of the segment config file.')
    parser.add_argument('--results_dir', required=True,
        help='Full directory to store results of segmentation.')
    parser.add_argument('--profiler_results_fullfname', required=True,
        help='Full file name (with path) of the profiler results text file.')
    parser.add_argument('--max_processes', required=True, type=int,
        help='The maximum number of processes to use.')
    parser.add_argument('--max_batch_memory', required=True, type=float,
        help='The maximum amount of memory in GB to use.')
    parser.add_argument('--use_gpu', required=False, default='no', choices=['yes', 'no'],
        help='Whether to use the GPU for computing features.')
    args = parser.parse_args()

    print('Running...')
    print()
    pr = cProfile.Profile()
    pr.enable()

    segment.main(
        segmenter=args.segmenter_fullfname,
        preproc_volume_fullfname=args.volume_fullfname,
        config=args.segment_config_fullfname,
        results_dir=args.results_dir,
        checkpoint_fullfname=None,
        checkpoint_namespace='segment',
        reset_checkpoint=False,
        checkpoint_init=dict(),
        max_processes=args.max_processes,
        max_batch_memory=args.max_batch_memory,
        use_gpu=args.use_gpu == 'yes',
        listener=listeners.CliProgressListener(),
        debug_mode=True
        )

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    with open(args.profiler_results_fullfname, 'w', encoding='utf-8') as f:
        lines = s.getvalue().strip().split('\n')
        lines[2] = lines[2].strip()
        for i in range(4, len(lines)):
            lines[i] = '\t'.join(lines[i].strip().split(None, 5))
        print('\n'.join(lines), file=f)


# main entry point
if __name__ == '__main__':
   main()
