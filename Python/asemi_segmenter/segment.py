'''Segment command.'''

import math
import os
import json
import pickle
import numpy as np
from asemi_segmenter.listener import ProgressListener
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import files
from asemi_segmenter.lib import images
from asemi_segmenter.lib import segmenters
from asemi_segmenter.lib import times
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        model, preproc_volume_fullfname, config, results_dir,
        checkpoint_fullfname, restart_checkpoint,
        listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Full volume data file')
    validations.check_filename(preproc_volume_fullfname, '.hdf', True)
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load()
    validations.validate_json_with_schema_file(full_volume.get_config(), 'preprocess.json')
    slice_shape = full_volume.get_shape()[1:]

    listener.log_output('> Model file')
    if isinstance(model, str):
        validations.check_filename(model, '.pkl', True)
        with open(model, 'rb') as f:
            pickled_data = pickle.load(f)
    else:
        pickled_data = model
    segmenter = segmenters.load_segmenter_from_pickle_data(pickled_data, full_volume, allow_random=False)

    listener.log_output('> Config file')
    if isinstance(config, str):
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    soft_segmentation = config_data['soft_segmentation'] == 'yes'

    listener.log_output('> Results directory')
    validations.check_directory(results_dir)

    listener.log_output('> Checkpoint file')
    if checkpoint_fullfname is not None:
        validations.check_filename(checkpoint_fullfname)
    checkpoint = checkpoints.CheckpointManager(
        'segment',
        checkpoint_fullfname,
        restart_checkpoint
        )

    listener.log_output('> Initialising')
    
    return (soft_segmentation, full_volume, slice_shape, segmenter, checkpoint)
    
    
#########################################
def _segmenting(
        soft_segmentation, full_volume, slice_shape, segmenter, results_dir, max_processes, max_batch_memory, checkpoint, listener
    ):
    '''Segmenting stage.'''
    for label in segmenter.classifier.labels:
        files.mkdir(os.path.join(results_dir, label))

    num_digits_in_filename = math.ceil(math.log10(full_volume.get_shape()[0]+1))
    best_block_shape = arrayprocs.get_optimal_block_size(
        slice_shape,
        full_volume.get_dtype(),
        segmenter.featuriser.get_context_needed(),
        max_processes,
        max_batch_memory,
        implicit_depth=True
        )

    with checkpoint.apply('segment') as skip:
        if skip is not None:
            raise skip
        start = checkpoint.get_next_to_process('segment_prog')
        listener.current_progress_start(start, full_volume.get_shape()[0])
        for volume_slice_index in range(full_volume.get_shape()[0]):
            if volume_slice_index < start:
                continue
            with checkpoint.apply('segment_prog'):
                slice_features = segmenter.featuriser.featurise_slice(
                    full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                    slice_index=volume_slice_index,
                    block_rows=best_block_shape[0],
                    block_cols=best_block_shape[1],
                    n_jobs=max_processes
                    )

                for (label, mask) in zip(segmenter.classifier.labels, segmenter.segment_to_labels_iter(slice_features, soft_segmentation, max_processes)):
                    mask = np.round(mask*255).astype(np.uint8)
                    mask = mask.reshape(slice_shape)
                    images.save_image(
                        os.path.join(
                            results_dir,
                            label,
                            '{}_{:0>{}}.tiff'.format(
                                label,
                                volume_slice_index+1,
                                num_digits_in_filename
                                )
                            ),
                        mask,
                        compress=True
                        )
            listener.current_progress_update(volume_slice_index+1)
        listener.current_progress_end()
        
    return ()


#########################################
def main(
        model, preproc_volume_fullfname, config, results_dir,
        checkpoint_fullfname, restart_checkpoint,
        max_processes, max_batch_memory, listener=ProgressListener()
    ):
    '''
    Segment a preprocessed volume using a trained classifier model.

    :param model: Full file name (with path) to saved pickle file or dictionary with model
        directly.
    :type model: str or dict
    :param str preproc_volume_fullfname: The full file name (with path) to the preprocessed
        volume HDF file.
    :param config: The configuration to use when segmenting (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the segment configuration.
    :type config: str or dict
    :param results_dir: The path to the directory in which to store the segmented slices. Segmented
        slices consist of a directory for each label, each containing images that act as masks for
        whether a particular pixel belongs to said label or not.
    :type results_dir: str
    :param str checkpoint_fullfname: Full file name (with path) to checkpoint pickle.
    :param checkpoint_fullfname: Full file name (with path) to checkpoint pickle. If None then no
        checkpointing is used.
    :type checkpoint_fullfname: str or None
    :param bool restart_checkpoint: Whether to ignore checkpoint and start process from beginning.
    :param int max_processes: The maximum number of processes to use concurrently.
    :param float max_batch_memory: The maximum number of gigabytes to use between all processes.
    :param ProgressListener listener: The command's progress listener.
    '''
    full_volume = None
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(2)

            listener.log_output('Starting segmentation process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (soft_segmentation, full_volume, slice_shape, segmenter, checkpoint) = _loading_data(
                    model, preproc_volume_fullfname, config, results_dir,
                    checkpoint_fullfname, restart_checkpoint,
                    listener
                    )
            listener.log_output('Data loaded')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Segmenting')
            listener.log_output(times.get_timestamp())
            listener.log_output('Segmenting')
            with times.Timer() as timer:
                () = _segmenting(soft_segmentation, full_volume, slice_shape, segmenter, results_dir, max_processes, max_batch_memory, checkpoint, listener)
            listener.log_output('Volume segmented')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()
    except Exception as ex:
        listener.error_output(str(ex))
    finally:
        if full_volume is not None:
            full_volume.close()
