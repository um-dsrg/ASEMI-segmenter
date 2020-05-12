'''Evaluate command.'''

import pickle
import os
import numpy as np
from asemi_segmenter.listener import ProgressListener
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import evaluations
from asemi_segmenter.lib import hashfunctions
from asemi_segmenter.lib import images
from asemi_segmenter.lib import results
from asemi_segmenter.lib import segmenters
from asemi_segmenter.lib import colours
from asemi_segmenter.lib import times
from asemi_segmenter.lib import files
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        segmenter, preproc_volume_fullfname, subvolume_dir, label_dirs, results_dir,
        checkpoint_fullfname, checkpoint_namespace, reset_checkpoint, checkpoint_init, max_processes, max_batch_memory, listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Volume')
    listener.log_output('>> {}'.format(preproc_volume_fullfname))
    validations.check_filename(preproc_volume_fullfname, '.hdf', True)
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load()
    preprocess_config = full_volume.get_config()
    validations.validate_json_with_schema_file(preprocess_config, 'preprocess.json')
    hash_function = hashfunctions.load_hashfunction_from_config(preprocess_config['hash_function'])
    slice_shape = full_volume.get_shape()[1:]
    slice_size = slice_shape[0]*slice_shape[1]

    listener.log_output('> Segmenter')
    if isinstance(segmenter, str):
        listener.log_output('>> {}'.format(segmenter))
        validations.check_filename(segmenter, '.pkl', True)
        with open(segmenter, 'rb') as f:
            pickled_data = pickle.load(f)
        segmenter = segmenters.load_segmenter_from_pickle_data(pickled_data, full_volume)

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
    evaluation_labels = sorted(label_data.name for label_data in labels_data)
    if evaluation_labels != segmenter.classifier.labels:
        raise ValueError('Labels in evaluation directory do not match labels in segmenter')
    validations.validate_annotation_data(full_volume, subvolume_data, labels_data)
    
    listener.log_output('> Result')
    listener.log_output('>> {}'.format(results_dir))
    validations.check_directory(results_dir)
    evaluation = evaluations.IntersectionOverUnionEvaluation(len(segmenter.classifier.labels))
    evaluation_results_file = results.EvaluationResultsFile(os.path.join(results_dir, 'results.txt'), evaluation)

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
    
    return (full_volume, slice_shape, slice_size, segmenter, subvolume_fullfnames, labels_data, hash_function, evaluation, evaluation_results_file, checkpoint)
    
    
#########################################
def _hashing_subvolume_slices(
        full_volume, subvolume_fullfnames, hash_function, listener
    ):
    '''Hashing subvolume slices stage.'''
    listener.current_progress_start(0, len(subvolume_fullfnames))
    subvolume_hashes = np.empty(
        (len(subvolume_fullfnames), hash_function.hash_size),
        full_volume.get_hashes_dtype()
        )
    for (i, fullfname) in enumerate(subvolume_fullfnames):
        img_data = images.load_image(fullfname)
        subvolume_hashes[i, :] = hash_function.apply(img_data)
        listener.current_progress_update(i+1)
    listener.current_progress_end()
    volume_slice_indexes_in_subvolume = volumes.get_volume_slice_indexes_in_subvolume(
        full_volume.get_hashes_array()[:], subvolume_hashes  #Load the hashes eagerly.
        )
    listener.log_output('> Subvolume to volume file name mapping found:')
    for (subvolume_index, volume_index) in enumerate(volume_slice_indexes_in_subvolume):
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
def _evaluating(
        full_volume, segmenter, slice_shape, slice_size, subvolume_fullfnames, volume_slice_indexes_in_subvolume, subvolume_slice_labels, evaluation, checkpoint, evaluation_results_file, results_dir, max_processes, max_batch_memory, listener
    ):
    '''Evaluating stage.'''
    listener.log_output('> Label sizes:')
    for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_subvolume):
        listener.log_output('>> Subvolume slice #{} (volume slice #{})'.format(i + 1, volume_slice_index + 1))
        for (label_index, label) in enumerate(segmenter.classifier.labels):
            listener.log_output('>>> {}: {}'.format(label, np.sum(subvolume_slice_labels[i*slice_size:(i+1)*slice_size] == label_index)))
    
    listener.log_output('> Evaluating')
    with checkpoint.apply('create_results_file') as skip:
        if skip is not None:
            listener.log_output('>> Continuing use of checkpointed results file')
            raise skip
        evaluation_results_file.create(segmenter.classifier.labels)
    best_block_shape = arrayprocs.get_optimal_block_size(
        slice_shape,
        full_volume.get_dtype(),
        segmenter.featuriser.get_context_needed(),
        max_processes,
        max_batch_memory,
        implicit_depth=True
        )
    
    labels_palette = colours.LabelPalette(['unlabelled'] + segmenter.classifier.labels)
    images.save_image(
        os.path.join(results_dir, 'labels_legend.tiff'),
        images.matplotlib_to_imagedata(labels_palette.get_legend()),
        num_bits=8,
        compress=True
        )
    confusion_map_saver = results.ConfusionMapSaver(segmenter.classifier.labels)
    images.save_image(
        os.path.join(results_dir, 'confusion_map_legend.tiff'),
        images.matplotlib_to_imagedata(confusion_map_saver.palette.get_legend()),
        num_bits=8,
        compress=True
        )
        
    start = checkpoint.get_next_to_process('evaluation_prog')
    listener.current_progress_start(start, len(subvolume_fullfnames))
    for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_subvolume):
        if i < start:
            continue
        with checkpoint.apply('evaluation_prog') as skip:
            with times.Timer() as sub_timer:
                with times.Timer() as sub_timer_featuriser:
                    slice_features = segmenter.featuriser.featurise_slice(
                        full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                        slice_index=volume_slice_index,
                        block_rows=best_block_shape[0],
                        block_cols=best_block_shape[1],
                        n_jobs=max_processes
                        )

                with times.Timer() as sub_timer_classifier:
                    prediction = segmenter.segment_to_label_indexes(slice_features, max_processes)
                
                slice_labels = subvolume_slice_labels[i*slice_size:(i+1)*slice_size]
                
                files.mkdir(os.path.join(results_dir, 'slice_{}'.format(i + 1)))
                
                reshaped_prediction = prediction.reshape(slice_shape)
                reshaped_slice_labels = slice_labels.reshape(slice_shape)
                
                images.save_image(
                    os.path.join(results_dir, 'slice_{}'.format(i + 1), 'input_slice.tiff'),
                    full_volume.get_scale_array(0)[volume_slice_index, :, :],
                    compress=True
                    )
                
                images.save_image(
                    os.path.join(results_dir, 'slice_{}'.format(i + 1), 'true_slice.tiff'),
                    labels_palette.label_indexes_to_colours(
                        np.where(
                            reshaped_slice_labels >= volumes.FIRST_CONTROL_LABEL,
                            0,
                            reshaped_slice_labels + 1
                            )
                        ),
                    num_bits=8,
                    compress=True
                    )
                
                images.save_image(
                    os.path.join(results_dir, 'slice_{}'.format(i + 1), 'predicted_slice.tiff'),
                    labels_palette.label_indexes_to_colours(
                        reshaped_prediction + 1
                        ),
                    num_bits=8,
                    compress=True
                    )
                
                confusion_matrix = evaluations.get_confusion_matrix(
                    reshaped_prediction,
                    reshaped_slice_labels,
                    len(segmenter.classifier.labels)
                    )
                results.save_confusion_matrix(
                    os.path.join(results_dir, 'slice_{}'.format(i + 1), 'confusion_matrix.txt'),
                    confusion_matrix,
                    segmenter.classifier.labels
                    )
                
                for (label_index, label) in enumerate(segmenter.classifier.labels):
                    confusion_map = evaluations.get_confusion_map(
                        reshaped_prediction,
                        reshaped_slice_labels,
                        label_index
                        )
                    confusion_map_saver.save(
                        os.path.join(results_dir, 'slice_{}'.format(i + 1), 'confusion_map_{}.tiff'.format(label)),
                        confusion_map
                        )
                
            evaluation_results_file.add(
                i + 1,
                volume_slice_index + 1,
                prediction,
                slice_labels,
                sub_timer_featuriser.duration,
                sub_timer_classifier.duration,
                sub_timer.duration
                )

        listener.current_progress_update(i+1)
    listener.current_progress_end()
    
    return ()


#########################################
def main(
        segmenter, preproc_volume_fullfname, subvolume_dir, label_dirs, results_dir,
        checkpoint_fullfname, checkpoint_namespace, reset_checkpoint, checkpoint_init,
        max_processes, max_batch_memory, listener=ProgressListener(), debug_mode=False
    ):
    '''
    Evaluate a trained segmenter on manually labelled slices.

    :param segmenter: Full file name (with path) to saved pickle file or segmenter object
        directly.
    :type segmenter: str or dict
    :param str preproc_volume_fullfname: The full file name (with path) to the preprocessed
        volume HDF file.
    :param str subvolume_dir: The path to the directory containing copies from the full
        volume slices that were labelled.
    :param list label_dirs: A list of paths to the directories containing labelled
        slices with the number of labels being equal to the number of directories and
        the number of images in each directory being equal to the number of subvolume
        images.
    :param str results_dir: The path to the directory to contain the results of this command.
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
    '''
    full_volume = None
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(4)

            listener.log_output('Starting evaluation process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (full_volume, slice_shape, slice_size, segmenter, subvolume_fullfnames, labels_data, hash_function, evaluation, evaluation_results_file, checkpoint) = _loading_data(
                    segmenter, preproc_volume_fullfname, subvolume_dir, label_dirs, results_dir,
                    checkpoint_fullfname, checkpoint_namespace, reset_checkpoint, checkpoint_init, max_processes, max_batch_memory,
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

            listener.overall_progress_update(4, 'Evaluating')
            listener.log_output(times.get_timestamp())
            listener.log_output('Evaluating')
            with times.Timer() as timer:
                () = _evaluating(full_volume, segmenter, slice_shape, slice_size, subvolume_fullfnames, volume_slice_indexes_in_subvolume, subvolume_slice_labels, evaluation, checkpoint, evaluation_results_file, results_dir, max_processes, max_batch_memory, listener)
            listener.log_output('Evaluated')
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