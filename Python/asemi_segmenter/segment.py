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

'''Segment command.'''

import math
import os
import json
import pickle
import numpy as np
from asemi_segmenter import listeners
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import files
from asemi_segmenter.lib import images
from asemi_segmenter.lib import segmenters
from asemi_segmenter.lib import times
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import featurisers


#########################################
def _loading_data(
        segmenter, preproc_volume_fullfname, config, results_dir, label_names_fullfname,
        slice_indexes, checkpoint_fullfname, checkpoint_namespace, reset_checkpoint,
        checkpoint_init, max_processes_featuriser, max_processes_classifier, max_batch_memory,
        num_simultaneous_slices, use_gpu, listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Volume')
    listener.log_output('>> {}'.format(preproc_volume_fullfname))
    validations.check_filename(preproc_volume_fullfname, '.hdf', True)
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load(as_readonly=True)
    validations.validate_json_with_schema_file(full_volume.get_config(), 'preprocess.json')
    slice_shape = full_volume.get_shape()[1:]

    listener.log_output('> Segmenter')
    if isinstance(segmenter, str):
        listener.log_output('>> {}'.format(segmenter))
        validations.check_filename(segmenter, '.pkl', True)
        with open(segmenter, 'rb') as f:
            pickled_data = pickle.load(f)
        segmenter = segmenters.load_segmenter_from_pickle_data(
            pickled_data,
            full_volume,
            max_batch_memory=max_batch_memory,
            use_gpu=use_gpu
            )

    listener.log_output('> Config')
    if isinstance(config, str):
        listener.log_output('>> {}'.format(config))
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    validations.validate_json_with_schema_file(config_data, 'segment.json')
    if config_data['soft_segmentation'] and not config_data['as_masks']:
        raise ValueError('Cannot use soft segmentation if output is not as masks.')

    listener.log_output('> Result')
    listener.log_output('>> {}'.format(results_dir))
    validations.check_directory(results_dir)

    listener.log_output('> Label names file')
    if label_names_fullfname is not None:
        listener.log_output('>> {}'.format(label_names_fullfname))
        validations.check_filename(label_names_fullfname, '.txt')

    listener.log_output('> Checkpoint')
    if checkpoint_fullfname is not None:
        listener.log_output('>> {}'.format(checkpoint_fullfname))
        validations.check_filename(checkpoint_fullfname)
    checkpoint = checkpoints.CheckpointManager(
        checkpoint_namespace,
        checkpoint_fullfname,
        reset_checkpoint=reset_checkpoint,
        initial_content=checkpoint_init
        )

    listener.log_output('> Calculating block shape')
    best_block_shape = arrayprocs.get_optimal_block_size(
        slice_shape,
        full_volume.get_dtype(),
        segmenter.featuriser.get_context_needed(),
        max_processes_featuriser,
        max_batch_memory,
        num_implicit_slices=num_simultaneous_slices,
        feature_size=segmenter.featuriser.get_feature_size(),
        feature_dtype=featurisers.feature_dtype
        )
    listener.log_output('>> Block shape: {}'.format(best_block_shape))
    listener.log_output('>> Block voxels memory usage: {:.3f}GB (out of {}GB)'.format(
        (2*segmenter.featuriser.get_context_needed() + num_simultaneous_slices)*np.prod(best_block_shape)*np.dtype(full_volume.get_dtype()).itemsize*max_processes_featuriser/(1024**3),
        max_batch_memory
        ))
    listener.log_output('>> Block features memory usage: {:.3f}GB (out of {}GB)'.format(
        num_simultaneous_slices*np.prod([l - 2*segmenter.featuriser.get_context_needed() for l in best_block_shape])*segmenter.featuriser.get_feature_size()*np.dtype(featurisers.feature_dtype).itemsize*max_processes_featuriser/(1024**3),
        max_batch_memory
        ))

    listener.log_output('> Initialising')

    listener.log_output('> Other parameters:')
    listener.log_output('>> reset_checkpoint: {}'.format(reset_checkpoint))
    listener.log_output('>> max_processes_featuriser: {}'.format(max_processes_featuriser))
    listener.log_output('>> max_processes_classifier: {}'.format(max_processes_classifier))
    listener.log_output('>> max_batch_memory: {}GB'.format(max_batch_memory))
    listener.log_output('>> num_simultaneous_slices: {}'.format(num_simultaneous_slices))
    if slice_indexes is not None:
        listener.log_output('>> slice_indexes: {}'.format(slice_indexes))
    if num_simultaneous_slices < 1:
        raise ValueError('num_simultaneous_slices must be a positive number.')
    if slice_indexes is not None and not isinstance(slice_indexes, range) and num_simultaneous_slices > 1:
        raise ValueError('num_simultaneous_slices can only be more than 1 when segmenting whole volume or range of slices.')
    if slice_indexes is None:
        slice_indexes = range(full_volume.get_shape()[0])
    elif isinstance(slice_indexes, range):
        slice_indexes = range(slice_indexes.start - 1, slice_indexes.stop)
    else:
        slice_indexes = sorted(i - 1 for i in set(slice_indexes))

    return (config_data, full_volume, slice_shape, slice_indexes, segmenter, best_block_shape, checkpoint)


#########################################
def _segmenting(
        config_data, full_volume, slice_shape, segmenter, results_dir, label_names_fullfname, slice_indexes, best_block_shape, max_processes_featuriser, max_processes_classifier, max_batch_memory, checkpoint, num_simultaneous_slices, listener
    ):
    '''Segmenting stage.'''
    if slice_indexes is None:
        num_slices = full_volume.get_shape()[0]
        last_slice_index = num_slices - 1
    elif isinstance(slice_indexes, range):
        num_slices = slice_indexes.stop - slice_indexes.start
        last_slice_index = slice_indexes.stop - 1
    else:
        num_slices = len(slice_indexes)
        last_slice_index = max(slice_indexes)

    slice_size = slice_shape[0]*slice_shape[1]

    if config_data['as_masks']:
        for label in segmenter.classifier.labels:
            files.mkdir(os.path.join(results_dir, label))

    if label_names_fullfname is not None:
        with open(label_names_fullfname, 'w', encoding='utf-8') as f:
            for label in segmenter.classifier.labels:
                print(label, file=f)

    num_digits_in_filename = math.ceil(math.log10(full_volume.get_shape()[0]+1))

    def save_slice(volume_slice_index, segmentation, label=None):
        '''Save image slice.'''
        if config_data['bits'] == 8:
            if config_data['as_masks']:
                segmentation = segmentation*(2**8 - 1)
            output = np.round(segmentation).astype(np.uint8)
        elif config_data['bits'] == 16:
            if config_data['as_masks']:
                segmentation = segmentation*(2**16 - 1)
            output = np.round(segmentation).astype(np.uint16)
        else:
            raise NotImplementedError('Number of bits not implemented.')
        if not config_data['as_masks']:
            output = output + 1
        output = output.reshape(slice_shape)

        images.save_image(
            os.path.join(
                results_dir if label is None else os.path.join(results_dir, label),
                '{}_{:0>{}}.{}'.format(
                    label if config_data['as_masks'] else 'seg',
                    volume_slice_index + 1,
                    num_digits_in_filename,
                    config_data['image_extension']
                    )
                ),
            output,
            num_bits=config_data['bits'],
            compress=True
            )

    with checkpoint.apply('segment') as skip:
        if skip is not None:
            listener.log_output('> Skipped as was found checkpointed')
            raise skip

        start = checkpoint.get_next_to_process('segment_prog')
        listener.current_progress_start(start, num_slices)
        for (i, first_volume_slice_index) in enumerate(slice_indexes):
            if i < start:
                continue
            with checkpoint.apply('segment_prog'):
                if (i - start)%num_simultaneous_slices == 0:
                    slice_features = segmenter.featuriser.featurise_slice(
                        full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                        slice_range=slice(first_volume_slice_index, min(first_volume_slice_index+num_simultaneous_slices, last_slice_index + 1)),
                        block_shape=best_block_shape,
                        max_processes=max_processes_featuriser
                        )
                    for j in range(num_simultaneous_slices):
                        if first_volume_slice_index + j > last_slice_index:
                            continue

                        if config_data['as_masks']:
                            for (label, mask) in zip(segmenter.classifier.labels, segmenter.segment_to_labels_iter(slice_features[j*slice_size:(j+1)*slice_size,:], config_data['soft_segmentation'], max_processes_classifier)):
                                save_slice(first_volume_slice_index + j, mask, label)
                        else:
                            save_slice(first_volume_slice_index + j, segmenter.segment_to_label_indexes(slice_features[j*slice_size:(j+1)*slice_size,:], max_processes_classifier))
            listener.current_progress_update(i+1)
        listener.current_progress_end()

    return ()


#########################################
def main(
        segmenter,
        preproc_volume_fullfname,
        config,
        results_dir,
        label_names_fullfname=None,
        slice_indexes=None,
        checkpoint_fullfname=None,
        checkpoint_namespace='segment',
        reset_checkpoint=False,
        checkpoint_init=dict(),
        max_processes_featuriser=-1,
        max_processes_classifier=-1,
        max_batch_memory=1,
        use_gpu=False,
        num_simultaneous_slices=1,
        listener=listeners.ProgressListener(),
        debug_mode=False
    ):
    '''
    Segment a preprocessed volume using a trained segmenter.

    :param segmenter: Full file name (with path) to saved segmenter pickle file or Segmenter object.
    :type segmenter: str or Segmenter
    :param str preproc_volume_fullfname: The full file name (with path) to the preprocessed
        volume HDF file.
    :param config: The configuration to use when segmenting (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the segment configuration.
    :type config: str or dict
    :param str results_dir: The path to the directory in which to store the segmented slices. Segmented
        slices consist of a directory for each label, each containing images that act as masks for
        whether a particular pixel belongs to said label or not.
    :param str label_names_fullfname: Full file name (with path) to text file in which to store
        label names.
    :param slice_indexes: The integer indexes (1-based) of slices in the volume to
        segment. If range then both start and stop are 1-based and inclusive. If
        None then all slices are segmented.
    :type slice_indexes: list or range or None
    :param str checkpoint_fullfname: Full file name (with path) to checkpoint pickle.
        If None then no checkpointing is used.
    :param str checkpoint_namespace: Namespace for the checkpoint file.
    :param bool reset_checkpoint: Whether to clear the checkpoint from the file (if it
        exists) and start afresh.
    :param dict checkpoint_init: The checkpoint data to initialise the checkpoint with,
        including the checkpoint file (only data about this particular command will be
        overwritten). If None then checkpoint is checkpoint file content if file exists,
        otherwise the checkpoint will be empty. To restart checkpoint set to empty dictionary.
    :param int max_processes_featuriser: The maximum number of processes to use concurrently
        whilst featurising.
    :param int max_processes_classifier: The maximum number of processes to use concurrently
        whilst classifying.
    :param float max_batch_memory: The maximum number of gigabytes to use between all featuriser
        processes.
    :param bool use_gpu: Whether to use the GPU for computing features. Note that this
        parameter does not do anything if the segmenter is provided directly.
    :param int num_simultaneous_slices: The number of adjacent slices to process together.
    :param ProgressListener listener: The command's progress listener.
    :param bool debug_mode: Whether to show full error messages or just simple ones.
    '''
    full_volume = None
    try:
        listener.overall_progress_start(2)

        listener.log_output('Starting segmentation process')
        listener.log_output('')

        with times.Timer() as full_timer:

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (config_data, full_volume, slice_shape, slice_indexes, segmenter, best_block_shape, checkpoint) = _loading_data(
                    segmenter, preproc_volume_fullfname, config, results_dir, label_names_fullfname,
                    slice_indexes, checkpoint_fullfname, checkpoint_namespace, reset_checkpoint,
                    checkpoint_init, max_processes_featuriser, max_processes_classifier, max_batch_memory, num_simultaneous_slices, use_gpu, listener
                    )
            listener.log_output('Data loaded')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            with checkpoint.apply('overall') as skip:
                if skip is not None:
                    listener.log_output('Command skipped as was found checkpointed')
                    raise skip

                ###################

                listener.overall_progress_update(2, 'Segmenting')
                listener.log_output(times.get_timestamp())
                listener.log_output('Segmenting')
                with times.Timer() as timer:
                    () = _segmenting(config_data, full_volume, slice_shape, segmenter, results_dir, label_names_fullfname, slice_indexes, best_block_shape, max_processes_featuriser, max_processes_classifier, max_batch_memory, checkpoint, num_simultaneous_slices, listener)
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
        if debug_mode:
            raise
    finally:
        if full_volume is not None:
            full_volume.close()
