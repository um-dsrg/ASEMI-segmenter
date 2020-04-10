'''Evaluate command.'''

import pickle
import numpy as np
from asemi_segmenter.listener import ProgressListener
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import evaluations
from asemi_segmenter.lib import hashfunctions
from asemi_segmenter.lib import images
from asemi_segmenter.lib import results
from asemi_segmenter.lib import segmenters
from asemi_segmenter.lib import times
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        model, preproc_volume_fullfname, subvolume_dir, label_dirs, results_fullfname,
        checkpoint_fullfname, restart_checkpoint, listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Full volume data file')
    validations.check_filename(preproc_volume_fullfname, '.hdf', True)
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load()
    preprocess_config = full_volume.get_config()
    validations.validate_json_with_schema_file(preprocess_config, 'preprocess.json')
    hash_function = hashfunctions.load_hashfunction_from_config(preprocess_config['hash_function'])
    slice_shape = full_volume.get_shape()[1:]
    slice_size = slice_shape[0]*slice_shape[1]

    listener.log_output('> Model file')
    if isinstance(model, str):
        validations.check_filename(model, '.pkl', True)
        with open(model, 'rb') as f:
            pickled_data = pickle.load(f)
    else:
        pickled_data = model
    segmenter = segmenters.load_segmenter_from_pickle_data(pickled_data, full_volume, allow_random=False)

    listener.log_output('> Labels')
    labels_data = []
    for (i, label_dir) in enumerate(label_dirs):
        listener.log_output('>> Loading label {} directory'.format(i+1))
        label_data = volumes.load_label_dir(label_dir)
        labels_data.append(label_data)
        listener.log_output('>>> {}'.format(label_data.name))
    evaluation_labels = sorted(label_data.name for label_data in labels_data)
    if evaluation_labels != segmenter.classifier.labels:
        raise ValueError('Labels in evaluation directory do not match labels in model')

    listener.log_output('> Subvolume directory')
    subvolume_data = volumes.load_volume_dir(subvolume_dir)
    validations.validate_annotation_data(full_volume, subvolume_data, labels_data)
    subvolume_fullfnames = subvolume_data.fullfnames
    
    listener.log_output('> Result file')
    if results_fullfname is not None:
        validations.check_filename(results_fullfname, '.txt', False)
    evaluation_results_file = results.EvaluationResultsFile(results_fullfname)

    listener.log_output('> Checkpoint file')
    if checkpoint_fullfname is not None:
        validations.check_filename(checkpoint_fullfname, '.json', False)
    checkpoint = checkpoints.CheckpointManager(
        'evaluate',
        checkpoint_fullfname,
        restart_checkpoint
        )
    
    listener.log_output('> Initialising')
    hash_function.init(slice_shape, seed=0)
    
    return (full_volume, slice_shape, slice_size, segmenter, subvolume_fullfnames, labels_data, hash_function, evaluation_results_file, checkpoint)
    
    
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
        full_volume, segmenter, slice_shape, slice_size, subvolume_fullfnames, volume_slice_indexes_in_subvolume, subvolume_slice_labels, checkpoint, evaluation_results_file, max_processes, max_batch_memory, listener
    ):
    '''Evaluating stage.'''
    output_result = dict()
    with checkpoint.apply('create_results_file') as skip:
        if skip is not None:
            listener.log_output('> Continuing use of checkpointed results file')
            listener.log_output('-')
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
    for (i, (subvolume_fullfname, volume_slice_index)) in enumerate(
            zip(subvolume_fullfnames, volume_slice_indexes_in_subvolume)
        ):
        listener.log_output('> Evaluating {} ({:.2%})'.format(
            subvolume_fullfname, (i+1)/len(subvolume_fullfnames)
            ))
        with checkpoint.apply('evaluating_{}'.format(volume_slice_index)) as skip:
            if skip is not None:
                listener.log_output('>> Skipped as was found checkpointed')
                listener.log_output('-')
                raise skip
            with times.Timer() as sub_timer:
                with times.Timer() as sub_timer_featuriser:
                    slice_features = segmenter.featuriser.featurise(
                        full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                        slice_index=volume_slice_index,
                        block_rows=best_block_shape[0],
                        block_cols=best_block_shape[1],
                        n_jobs=max_processes
                        )

                with times.Timer() as sub_timer_classifier:
                    prediction = segmenter.segment_to_label_indexes(slice_features, max_processes)
                prediction = prediction.reshape(slice_shape)

                slice_labels = subvolume_slice_labels[i*slice_size:(i+1)*slice_size]
                slice_labels = slice_labels.reshape(slice_shape)

                ious = evaluations.get_intersection_over_union(
                    prediction, slice_labels, len(segmenter.classifier.labels)
                    )
                for (label, iou) in zip(segmenter.classifier.labels, ious):
                    if iou is not None:
                        listener.log_output('>> {}: {:.3%}'.format(label, iou))
                evaluation_results_file.append(
                    subvolume_fullfname, ious,
                    sub_timer_featuriser.duration, sub_timer_classifier.duration
                    )
                output_result[subvolume_fullfname] = ious

            listener.log_output('   Duration: {} (featurisation: {}, ' \
                'classification: {})'.format(
                    times.get_readable_duration(sub_timer.duration),
                    times.get_readable_duration(sub_timer_featuriser.duration),
                    times.get_readable_duration(sub_timer_classifier.duration)
                    ))
            listener.log_output('-')
            
    return (output_result,)



#########################################
def main(
        model, preproc_volume_fullfname, subvolume_dir, label_dirs, results_fullfname,
        checkpoint_fullfname, restart_checkpoint, max_processes, max_batch_memory,
        listener=ProgressListener()
    ):
    '''
    Evaluate a trained classifier model on manually labelled slices.

    :param model: Full file name (with path) to saved pickle file or dictionary with model
        directly.
    :type model: str or dict
    :param str preproc_volume_fullfname: The full file name (with path) to the preprocessed
        volume HDF file.
    :param str subvolume_dir: The path to the directory containing copies from the full
        volume slices that were labelled.
    :param list label_dirs: A list of paths to the directories containing labelled
        slices with the number of labels being equal to the number of directories and
        the number of images in each directory being equal to the number of subvolume
        images.
    :param results_fullfname: The full file name (with path) to the results text file to save.
        Results consist of a table of intersection-over-union scores and durations for each slice in
        subvolume_dir. If set to None then result will not be saved but will instead be returned
        as a dictionary of subvolume slice paths mapped to their intersection-over-union scores.
    :type results_fullfname: str or None
    :param str checkpoint_fullfname: Full file name (with path) to checkpoint pickle.
    :param checkpoint_fullfname: Full file name (with path) to checkpoint pickle. If None then no
        checkpointing is used.
    :type checkpoint_fullfname: str or None
    :param bool restart_checkpoint: Whether to ignore checkpoint and start process from beginning.
    :param int max_processes: The maximum number of processes to use concurrently.
    :param float max_batch_memory: The maximum number of gigabytes to use between all processes.
    :param ProgressListener listener: The command's progress listener.
    :return: If results_fullfname was None, returns the results as a dictionary of subvolume slice
        paths mapped to their intersection-over-union scores.
    :rtype: None or dict
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
                (full_volume, slice_shape, slice_size, segmenter, subvolume_fullfnames, labels_data, hash_function, evaluation_results_file, checkpoint) = _loading_data(
                    model, preproc_volume_fullfname, subvolume_dir, label_dirs, results_fullfname,
                    checkpoint_fullfname, restart_checkpoint, listener
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
            listener.log_output('-')
            with times.Timer() as timer:
                (output_result,) = _evaluating(full_volume, segmenter, slice_shape, slice_size, subvolume_fullfnames, volume_slice_indexes_in_subvolume, subvolume_slice_labels, checkpoint, evaluation_results_file, max_processes, max_batch_memory, listener)
            listener.log_output('Evaluated')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()

        if results_fullfname is None:
            return output_result
        return None
    except Exception as ex:
        listener.error_output(str(ex))
    finally:
        if full_volume is not None:
            full_volume.close()