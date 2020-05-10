'''train command.'''

import json
import numpy as np
from asemi_segmenter.listener import ProgressListener
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import hashfunctions
from asemi_segmenter.lib import images
from asemi_segmenter.lib import segmenters
from asemi_segmenter.lib import times
from asemi_segmenter.lib import datasets
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        preproc_volume_fullfname, subvolume_dir, label_dirs, config,
        result_segmenter_fullfname, trainingset_file_fullfname,
        checkpoint_fullfname, checkpoint_namespace, reset_checkpoint,
        checkpoint_init, max_processes, max_batch_memory, listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Volume')
    listener.log_output('>> {}'.format(preproc_volume_fullfname))
    validations.check_filename(preproc_volume_fullfname, '.hdf', False)
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load()
    preprocess_config = full_volume.get_config()
    validations.validate_json_with_schema_file(preprocess_config, 'preprocess.json')
    hash_function = hashfunctions.load_hashfunction_from_config(preprocess_config['hash_function'])
    slice_shape = full_volume.get_shape()[1:]
    slice_size = slice_shape[0]*slice_shape[1]
    
    listener.log_output('> Subvolume')
    listener.log_output('>> {}'.format(subvolume_dir))
    subvolume_data = volumes.load_volume_dir(subvolume_dir)
    subvolume_fullfnames = subvolume_data.fullfnames

    listener.log_output('> Labels')
    labels_data = []
    for label_dir in label_dirs:
        listener.log_output('>> {}'.format(label_dir))
        label_data = volumes.load_label_dir(label_dir)
        labels_data.append(label_data)
        listener.log_output('>>> {}'.format(label_data.name))
    validations.validate_annotation_data(full_volume, subvolume_data, labels_data)
    labels = sorted(label_data.name for label_data in labels_data)

    listener.log_output('> Config')
    if isinstance(config, str):
        listener.log_output('>> {}'.format(config))
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    segmenter = segmenters.Segmenter(labels, full_volume, config_data, as_samples=False)
    if 'samples_to_skip_per_label' not in config_data['training_set']:
        config_data['training_set']['samples_to_skip_per_label'] = 0

    listener.log_output('> Result')
    if result_segmenter_fullfname is not None:
        listener.log_output('>> {}'.format(result_segmenter_fullfname))
        validations.check_filename(result_segmenter_fullfname, '.pkl', False)
    
    listener.log_output('> Training set')
    if trainingset_file_fullfname is not None:
        listener.log_output('>> {}'.format(trainingset_file_fullfname))
        validations.check_filename(trainingset_file_fullfname, '.hdf', False)
    training_set = datasets.DataSet(trainingset_file_fullfname)

    listener.log_output('> Checkpoint')
    if checkpoint_fullfname is not None:
        listener.log_output('>> {}'.format(checkpoint_fullfname))
        validations.check_filename(checkpoint_fullfname, '.json', False)
    checkpoint = checkpoints.CheckpointManager(
        checkpoint_namespace,
        checkpoint_fullfname,
        reset_checkpoint=reset_checkpoint,
        initial_content=checkpoint_init
        )
        
    listener.log_output('> Initialising')
    hash_function.init(slice_shape, seed=0)
    
    listener.log_output('> Other parameters:')
    listener.log_output('>> reset_checkpoint: {}'.format(reset_checkpoint))
    listener.log_output('>> max_processes: {}'.format(max_processes))
    listener.log_output('>> max_batch_memory: {}GB'.format(max_batch_memory))
    
    return (full_volume, subvolume_fullfnames, labels_data, slice_shape, slice_size, segmenter, training_set, hash_function, checkpoint)


#########################################
def _hashing_subvolume_slices(
        full_volume, subvolume_fullfnames, hash_function, listener
    ):
    '''Hashing subvolume slices stage.'''
    listener.current_progress_start(0, len(subvolume_fullfnames))
    subvolume_hashes = np.empty(
        (len(subvolume_fullfnames), hash_function.hash_size),
        full_volume.get_hashes_dtype())
    for (i, fullfname) in enumerate(subvolume_fullfnames):
        img_data = images.load_image(fullfname)
        subvolume_hashes[i, :] = hash_function.apply(img_data)
        listener.current_progress_update(i+1)
    listener.current_progress_end()
    volume_slice_indexes_in_subvolume = volumes.get_volume_slice_indexes_in_subvolume(
        full_volume.get_hashes_array()[:], subvolume_hashes  #Load the hashes eagerly.
        )
    listener.log_output('> Subvolume to volume file name mapping found:')
    for (subvolume_index, volume_index) in enumerate(
            volume_slice_indexes_in_subvolume
        ):
        listener.log_output('>> {} -> volume slice #{}'.format(
            subvolume_fullfnames[subvolume_index], volume_index+1
            ))

    return (volume_slice_indexes_in_subvolume,)


#########################################
def _constructing_labels_dataset(
        labels_data
    ):
    '''Constructing labels dataset stage.'''
    subvolume_slice_labels = volumes.load_labels(labels_data)
    
    return (subvolume_slice_labels,)


#########################################
def _constructing_trainingset(
        full_volume, subvolume_fullfnames, volume_slice_indexes_in_subvolume, slice_shape, slice_size, subvolume_slice_labels, segmenter, training_set, checkpoint, max_processes, max_batch_memory, listener
    ):
    '''Constructing training set stage.'''
    listener.log_output('> Sampling training items')
    sample_size_per_label = segmenter.train_config['training_set']['sample_size_per_label']
    if sample_size_per_label != -1:
        (voxel_indexes, label_positions) = datasets.sample_voxels(
            subvolume_slice_labels,
            sample_size_per_label,
            len(segmenter.classifier.labels),
            volume_slice_indexes_in_subvolume,
            slice_shape,
            segmenter.train_config['training_set']['samples_to_skip_per_label'],
            seed=0
            )
    
    listener.log_output('> Train label sizes:')
    if sample_size_per_label != -1:
        for (label, label_slice) in zip(segmenter.classifier.labels, label_positions):
            listener.log_output('>> {}: {}'.format(label, label_slice.stop - label_slice.start))
    else:
        for (label_index, label) in enumerate(segmenter.classifier.labels):
            listener.log_output('>> {}: {}'.format(label, np.sum(subvolume_slice_labels == label_index)))
    
    listener.log_output('> Creating empty training set')
    if sample_size_per_label != -1:
        training_set.create(
            len(voxel_indexes),
            segmenter.featuriser.get_feature_size()
            )
    else:
        training_set.create(
            subvolume_slice_labels.size,
            segmenter.featuriser.get_feature_size()
            )
    training_set.load()
    
    listener.log_output('> Constructing labels')
    if sample_size_per_label != -1:
        for (label_index, label_position) in enumerate(label_positions):
            training_set.get_labels_array()[label_position] = label_index
    else:
        with checkpoint.apply('contructing_labels') as skip:
            if skip is not None:
                listener.log_output('> Skipped as was found checkpointed')
                raise skip
            training_set.get_labels_array()[:] = subvolume_slice_labels

    listener.log_output('> Constructing features')
    if sample_size_per_label != -1:
        segmenter.featuriser.featurise_voxels(
            full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
            voxel_indexes,
            output=training_set.get_features_array(),
            n_jobs=max_processes
            )
    else:
        best_block_shape = arrayprocs.get_optimal_block_size(
            slice_shape,
            full_volume.get_dtype(),
            segmenter.featuriser.get_context_needed(),
            max_processes,
            max_batch_memory,
            implicit_depth=True
            )
        with checkpoint.apply('constructing_features') as skip:
            if skip is not None:
                listener.log_output('> Skipped as was found checkpointed')
                raise skip
            start = checkpoint.get_next_to_process('constructing_features_prog')
            listener.current_progress_start(start, len(subvolume_fullfnames))
            for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_subvolume):
                if i < start:
                    continue
                with checkpoint.apply('constructing_features_prog'):
                    segmenter.featuriser.featurise_slice(
                        full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                        slice_index=volume_slice_index,
                        block_rows=best_block_shape[0],
                        block_cols=best_block_shape[1],
                        output=training_set.get_features_array(),
                        output_start_row_index=i*slice_size,
                        n_jobs=max_processes
                        )
                listener.current_progress_update(i+1)
            listener.current_progress_end()
            
    return ()


#########################################
def _training_segmenter(
        segmenter, training_set, max_processes
    ):
    '''Training segmenter stage.'''
    sample_size_per_label = segmenter.train_config['training_set']['sample_size_per_label']
    if sample_size_per_label == -1:
        training_set = training_set.without_control_labels()
    segmenter.train(training_set, max_processes)
    
    return ()


#########################################
def _saving_segmenter(
        segmenter, result_segmenter_fullfname, listener
    ):
    '''Saving segmenter stage.'''
    if result_segmenter_fullfname is not None:
        segmenter.save(result_segmenter_fullfname)
    else:
        listener.log_output('Segmenter not to be saved')
    
    return ()


#########################################
def main(
        preproc_volume_fullfname, subvolume_dir, label_dirs, config,
        result_segmenter_fullfname, checkpoint_namespace, trainingset_file_fullfname,
        checkpoint_fullfname, reset_checkpoint, checkpoint_init,
        max_processes, max_batch_memory, listener=ProgressListener(),
        debug_mode=False
    ):
    '''
    Train a segmenter to segment volumes based on manually labelled slices.

    :param str preproc_volume_fullfname: The full file name (with path) to the preprocessed
        volume HDF file.
    :param str subvolume_dir: The path to the directory containing copies from the full
        volume slices that were labelled.
    :param list label_dirs: A list of paths to the directories containing labelled
        slices with the number of labels being equal to the number of directories and
        the number of images in each directory being equal to the number of subvolume
        images.
    :param config: The configuration to use when training (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the train configuration.
    :type config: str or dict
    :param result_segmenter_fullfname: Full file name (with path) to pickle file to create. If None
        then it will not be saved.
    :type result_segmenter_fullfname: str or None
    :param str checkpoint_fullfname: Full file name (with path) to checkpoint pickle.
        If None then no checkpointing is used.
    :param str checkpoint_namespace: Namespace for the checkpoint file.
    :param bool reset_checkpoint: Whether to clear the checkpoint from the file (if it
        exists) and start afresh.
    :param dict checkpoint_init: The checkpoint data to initialise the checkpoint with,
        including the checkpoint file (only data about this particular command will be
        overwritten). If None then checkpoint is checkpoint file content if file exists,
        otherwise the checkpoint will be empty. To restart checkpoint set to empty dictionary.
    :param int max_processes: The maximum number of processes to use concurrently.
    :param float max_batch_memory: The maximum number of gigabytes to use between all processes.
    :param ProgressListener listener: The command's progress listener.
    :param bool debug_mode: Whether to show full error messages or just simple ones.
    :return: The segmenter object.
    :rtype: segmenters.Segmenter
    '''
    full_volume = None
    training_set = None
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(6)

            listener.log_output('Starting training process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (full_volume, subvolume_fullfnames, labels_data, slice_shape, slice_size, segmenter, training_set, hash_function, checkpoint) = _loading_data(
                    preproc_volume_fullfname, subvolume_dir, label_dirs, config,
                    result_segmenter_fullfname, trainingset_file_fullfname,
                    checkpoint_fullfname, checkpoint_namespace, reset_checkpoint,
                    checkpoint_init, max_processes, max_batch_memory,
                    listener
                    )
            listener.log_output('Data loaded')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Hashing subvolume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing subvolume slices')
            with times.Timer() as timer:
                (volume_slice_indexes_in_subvolume,) = _hashing_subvolume_slices(full_volume, subvolume_fullfnames, hash_function, listener)
            listener.log_output('Slices hashed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(3, 'Constructing labels dataset')
            listener.log_output(times.get_timestamp())
            listener.log_output('Constructing labels dataset')
            with times.Timer() as timer:
                (subvolume_slice_labels,) = _constructing_labels_dataset(labels_data)
            listener.log_output('Labels dataset constructed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')
            
            ###################

            listener.overall_progress_update(4, 'Constructing training set')
            listener.log_output(times.get_timestamp())
            listener.log_output('Constructing training set')
            with times.Timer() as timer:
                () = _constructing_trainingset(full_volume, subvolume_fullfnames, volume_slice_indexes_in_subvolume, slice_shape, slice_size, subvolume_slice_labels, segmenter, training_set, checkpoint, max_processes, max_batch_memory, listener)
            listener.log_output('Training set constructed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(5, 'Training segmenter')
            listener.log_output(times.get_timestamp())
            listener.log_output('Training segmenter')
            with times.Timer() as timer:
                () = _training_segmenter(segmenter, training_set, max_processes)
            listener.log_output('Segmenter trained')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(6, 'Saving segmenter')
            listener.log_output(times.get_timestamp())
            listener.log_output('Saving segmenter')
            with times.Timer() as timer:
                () = _saving_segmenter(segmenter, result_segmenter_fullfname, listener)
            listener.log_output('Segmenter saved')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()

        return segmenter
    except Exception as ex:
        listener.error_output(str(ex))
        if debug_mode:
            raise
    finally:
        if full_volume is not None:
            full_volume.close()
        if training_set is not None and trainingset_file_fullfname is not None:
            training_set.close()