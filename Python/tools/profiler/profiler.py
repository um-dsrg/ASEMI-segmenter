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
    parser.add_argument('--max_processes_featuriser', required=True, type=int,
        help='The maximum number of processes to use whilst featurising.')
    parser.add_argument('--max_processes_classifier', required=True, type=int,
        help='The maximum number of processes to use whilst classifying.')
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
        max_processes_featuriser=args.max_processes_featuriser,
        max_processes_classifier=args.max_processes_classifier,
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
