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
import os
import sys
import socket
import psutil
import asemi_segmenter
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter import train
from asemi_segmenter import listeners

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Train a new segmenter.'
        )

    parser.add_argument(
            '--preproc_volume_fullfname',
            required=True,
            help='Full path to preprocessed volume data file (*.hdf).'
        )
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
            help='Full path to configuration file specifying how to extract features and classify them (*.json).'
        )
    parser.add_argument(
            '--result_segmenter_fullfname',
            required=True,
            help='Full path to segmenter pickle file to be created by this process (*.pkl).'
        )
    parser.add_argument(
            '--trainingset_file_fullfname',
            required=False,
            default=None,
            help='Full path to file that is used to store the training set (*.hdf). Note that if this is left out then there is nothing to checkpoint.'
        )
    parser.add_argument(
            '--verbose_training',
            required=False,
            default='yes',
            choices=['yes', 'no'],
            help='Whether to show sklearn\'s verbose messages during training (default is yes).'
        )
    parser.add_argument(
            '--train_sample_seed',
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
            '--checkpoint_namespace',
            required=False,
            default='train',
            help='Unique name for the group of checkpoints used by this command.'
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
            '--max_processes_featuriser',
            required=False,
            default=-1,
            type=int,
            help='Maximum number of parallel processes to use concurrently whilst featurising (-1 to use maximum, default).'
        )
    parser.add_argument(
            '--max_processes_classifier',
            required=False,
            default=-1,
            type=int,
            help='Maximum number of parallel processes to use concurrently whilst classifying (-1 to use maximum, default).'
        )
    parser.add_argument(
            '--max_batch_memory',
            required=False,
            default=-1,
            type=float,
            help='Maximum amount of GB to allow for processing the volume in batches (-1 to use maximum, default).'
        )
    parser.add_argument(
            '--use_gpu',
            required=False,
            default='no',
            choices=['yes', 'no'],
            help='Whether to use the GPU for computing features (default is no).'
        )
    parser.add_argument(
            '--print_output',
            required=False,
            default='yes',
            choices=['yes', 'no'],
            help='Whether to output to the screen (default is yes).'
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
        listener = listeners.CliProgressListener(args.log_file_fullfname, print_output=args.print_output == 'yes')
        listener.log_output('='*100)
        listener.log_output('version:  {}'.format(asemi_segmenter.__version__))
        listener.log_output('hostname: {}'.format(socket.gethostname()))
        listener.log_output('bin dir:  {}'.format(os.path.dirname(os.path.realpath(__file__))))
        listener.log_output('curr dir: {}'.format(os.getcwd()))
        listener.log_output('-'*50)
        listener.log_output('')
        train.main(
            preproc_volume_fullfname=args.preproc_volume_fullfname,
            subvolume_dir=args.subvolume_dir,
            label_dirs=args.label_dirs,
            config=args.config_fullfname,
            result_segmenter_fullfname=args.result_segmenter_fullfname,
            trainingset_file_fullfname=args.trainingset_file_fullfname,
            verbose_training=args.verbose_training == 'yes',
            train_sample_seed=args.train_sample_seed,
            checkpoint_fullfname=args.checkpoint_fullfname,
            checkpoint_namespace=args.checkpoint_namespace,
            reset_checkpoint=args.reset_checkpoint == 'yes',
            checkpoint_init=dict(),
            max_processes_featuriser=arrayprocs.get_num_processes(args.max_processes_featuriser),
            max_processes_classifier=arrayprocs.get_num_processes(args.max_processes_classifier),
            max_batch_memory=(
                args.max_batch_memory
                if args.max_batch_memory > 0
                else psutil.virtual_memory().available/(1024**3)
                ),
            use_gpu=args.use_gpu == 'yes',
            listener=listener,
            debug_mode=args.debug_mode == 'yes'
            )
        listener.log_output('')
        listener.log_output('')
