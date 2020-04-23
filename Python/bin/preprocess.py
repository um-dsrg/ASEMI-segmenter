import argparse
import os
import sys
import socket
import psutil
import _interfaces
import asemi_segmenter
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter import preprocess

if __name__ == '__main__':
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

    args = None
    args = parser.parse_args()
    if args is not None:
        listener = _interfaces.CliProgressListener(args.log_file_fullfname)
        listener.log_output('='*100)
        listener.log_output('version: {}'.format(asemi_segmenter.__version__))
        listener.log_output('hostname: {}'.format(socket.gethostname()))
        listener.log_output('bin dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        listener.log_output('-'*50)
        listener.log_output('')
        preprocess.main(
            volume_dir=args.volume_dir,
            config=args.config_fullfname,
            result_data_fullfname=args.result_data_fullfname,
            checkpoint_fullfname=args.checkpoint_fullfname,
            restart_checkpoint=args.restart_checkpoint == 'yes',
            max_processes=arrayprocs.get_num_processes(args.max_processes),
            max_batch_memory=(
                args.max_batch_memory
                if args.max_batch_memory > 0
                else psutil.virtual_memory().available/(1024**3)
                ),
            listener=listener
            )
        listener.log_output('')
        listener.log_output('')