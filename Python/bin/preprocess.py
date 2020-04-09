import argparse
import os
import sys
import psutil
import _interfaces
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter import preprocess

parser = argparse.ArgumentParser(
        description='Preprocess a folder of slice images into a single volume file that can be used by the other commands.'
    )

parser.add_argument(
        '--volume_dir',
        required=True,
        help='Directory of full volume of image slices.'
    )
parser.add_argument(
        '--config_fullfname',
        required=True,
        help='Full path to configuration file specifying how to preprocess the volume (*.json).'
    )
parser.add_argument(
        '--result_data_fullfname',
        required=True,
        help='Full path of data file to store the preprocessed volume.'
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
    if True:
        assert 'config' not in params
        params['config'] = params['config_fullfname']
        del params['config_fullfname']
    if 'restart_checkpoint' in params:
        params['restart_checkpoint'] = params['restart_checkpoint'] == 'yes'
    if 'max_processes' in params:
        params['max_processes'] = arrayprocs.get_num_processes(params['max_processes'])
    if 'max_batch_memory' in params:
        if params['max_batch_memory'] <= 0:
            params['max_batch_memory'] = psutil.virtual_memory().available/(1024**3)
    params['listener'].log_output('='*100)
    preprocess.main(**params)
    params['listener'].log_output('')
    params['listener'].log_output('')