import argparse
import os
import sys
import psutil
import _interfaces
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter import evaluate

parser = argparse.ArgumentParser(
        description='Evaluate the performance of a trained segmentation model using the intersection-over-union metric.'
    )

#########################

parser.add_argument(
        '--model_fullfname',
        required=True,
        help='Full path to model pickle file that was obtained from train command (*.pkl).'
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
        '--results_fullfname',
        required=False,
        help='Full path to results text file to be created by this process (*.txt). If left out then results will just be displayed.'
    )
parser.add_argument(
        '--checkpoint_fullfname',
        required=False,
        default=None,
        help='Full path to file that is used to let the process save its progress and continue from where it left off in case of interruption (*.pkl). If left out then the process will run from beginning to end without saving any checkpoints.'
    )
parser.add_argument(
        '--restart_checkpoint',
        required=False,
        default='no',
        choices=['yes', 'no'],
        help='Whether to skip what has already been completed according to the saved checkpoint file or not (default is no).'
    )
parser.add_argument(
        '--log_file_fullfname',
        required=False,
        default=None,
        help='Full path to file that is used to store a log of what is displayed on screen (*.txt).'
    )
parser.add_argument(
        '--max_processes',
        required=False,
        default=-1,
        type=int,
        help='Maximum number of parallel processes to use (-1 to use maximum, default).'
    )
parser.add_argument(
        '--max_batch_memory',
        required=False,
        default=-1,
        type=float,
        help='Maximum amount of GB to allow for processing the volume in batches (-1 to use maximum, default).'
    )

#########################

params = None
params = vars(parser.parse_args())
if params is not None:
    params['listener'] = _interfaces.CliProgressListener(params.pop('log_file_fullfname'))
    if 'model_fullfname' in params:
        assert 'model' not in params
        params['model'] = params['model_fullfname']
        del params['model_fullfname']
    if 'restart_checkpoint' in params:
        params['restart_checkpoint'] = params['restart_checkpoint'] == 'yes'
    if 'max_processes' in params:
        params['max_processes'] = arrayprocs.get_num_processes(params['max_processes'])
    if 'max_batch_memory' in params:
        if params['max_batch_memory'] <= 0:
            params['max_batch_memory'] = psutil.virtual_memory().available/(1024**3)
    params['listener'].log_output('='*100)
    evaluate.main(**params)
    params['listener'].log_output('')
    params['listener'].log_output('')