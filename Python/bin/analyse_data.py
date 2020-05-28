#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.

import argparse
import os
import socket
import asemi_segmenter
from asemi_segmenter import analyse_data
from asemi_segmenter import listeners

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Analyse a dataset.'
        )

    #########################

    parser.add_argument(
            '--subvolume_dir',
            required=True,
            help='Directory of subvolume that was labelled.'
        )
    parser.add_argument(
            '--label_dirs',
            nargs='+',
            required=True,
            help='Directories of labels containing the slices in subvolume_dir with unlabelled regions blacked out.'
        )
    parser.add_argument(
            '--config_fullfname',
            required=True,
            help='Full path to configuration file specifying how to extract the dataset (*.json).'
        )
    parser.add_argument(
            '--highlight_radius',
            required=True,
            type=int,
            help='The radius of the squares that mark the sampled voxels in the slices.'
        )
    parser.add_argument(
            '--results_dir',
            required=True,
            help='Directory of folder to contain results.'
        )
    parser.add_argument(
            '--data_sample_seed',
            required=False,
            default=None,
            type=int,
            help='Seed for the random number generator which samples voxels. If left out then the random number generator will be non-deterministic.'
        )
    parser.add_argument(
            '--checkpoint_fullfname',
            required=False,
            default=None,
            help='Full path to file that is used to let the process save its progress and continue from where it left off in case of interruption (*.pkl). If left out then the process will run from beginning to end without saving any checkpoints.'
        )
    parser.add_argument(
            '--reset_checkpoint',
            required=False,
            default='no',
            choices=['yes', 'no'],
            help='Whether to clear the checkpoints about this command from the checkpoint file and start afresh or not (default is no).'
        )
    parser.add_argument(
            '--log_file_fullfname',
            required=False,
            default=None,
            help='Full path to file that is used to store a log of what is displayed on screen (*.txt).'
        )
    parser.add_argument(
            '--debug_mode',
            required=False,
            default='no',
            choices=['yes', 'no'],
            help='Whether to give full error messages (default is no).'
        )

    #########################

    args = None
    args = parser.parse_args()
    if args is not None:
        listener = listeners.CliProgressListener(args.log_file_fullfname)
        listener.log_output('='*100)
        listener.log_output('version: {}'.format(asemi_segmenter.__version__))
        listener.log_output('hostname: {}'.format(socket.gethostname()))
        listener.log_output('bin dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        listener.log_output('-'*50)
        listener.log_output('')
        analyse_data.main(
            subvolume_dir=args.subvolume_dir,
            label_dirs=args.label_dirs,
            config=args.config_fullfname,
            highlight_radius=args.highlight_radius,
            results_dir=args.results_dir,
            data_sample_seed=args.data_sample_seed,
            checkpoint_fullfname=args.checkpoint_fullfname,
            checkpoint_namespace='analyse_data',
            reset_checkpoint=args.reset_checkpoint == 'yes',
            checkpoint_init=dict(),
            listener=listener,
            debug_mode=args.debug_mode == 'yes'
            )
        listener.log_output('')
        listener.log_output('')
