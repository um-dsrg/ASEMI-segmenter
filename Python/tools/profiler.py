#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti

import argparse
import cProfile
import pstats
import os
import io
import textwrap
import tqdm
from asemi_segmenter import listener
from asemi_segmenter import segment

TEXT_WIDTH = 100

#########################################
class ProgressListener(listener.ProgressListener):

    #########################################
    def __init__(self):
        self.prog = None
        self.prog_prev_value = 0

    #########################################
    def log_output(self, text):
        if text == '':
            print()
        else:
            for (i, line) in enumerate(textwrap.wrap(text, TEXT_WIDTH)):
                if i == 0:
                    print(line)
                else:
                    print('   '+line)

    #########################################
    def current_progress_start(self, start, total):
        self.prog = tqdm.tqdm(initial=start, total=total)
        self.prog_prev_value = start

    #########################################
    def current_progress_update(self, curr):
        self.prog.update(curr - self.prog_prev_value)
        self.prog_prev_value = curr

    #########################################
    def current_progress_end(self):
        self.prog.close()

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
        listener=ProgressListener(),
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
