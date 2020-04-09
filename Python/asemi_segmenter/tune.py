'''Tune command.'''

import json
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
from asemi_segmenter.lib import trainingsets
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
        eval_subvolume_dir, eval_label_dirs, config,
        results_fullfname, checkpoint_fullfname, restart_checkpoint,
        listener
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
    
    listener.log_output('> Train subvolume directory')
    train_subvolume_data = volumes.load_volume_dir(train_subvolume_dir)
    train_subvolume_fullfnames = train_subvolume_data.fullfnames

    listener.log_output('> Train labels')
    train_labels_data = []
    for label_dir in train_label_dirs:
        listener.log_output('>> {}'.format(label_dir))
        label_data = volumes.load_label_dir(label_dir)
        train_labels_data.append(label_data)
        listener.log_output('>>> {}'.format(label_data.name))
    train_labels = sorted(label_data.name for label_data in train_labels_data)
    train_subvolume_slice_labels = volumes.load_labels(train_labels_data)
    validations.validate_annotation_data(full_volume, train_subvolume_data, train_labels_data)

    listener.log_output('> Eval subvolume directory')
    eval_subvolume_data = volumes.load_volume_dir(eval_subvolume_dir)

    listener.log_output('> Eval labels')
    eval_labels_data = []
    for label_dir in eval_label_dirs:
        listener.log_output('>> {}'.format(label_dir))
        label_data = volumes.load_label_dir(label_dir)
        eval_labels_data.append(label_data)
        listener.log_output('>>> {}'.format(label_data.name))
    eval_subvolume_fullfnames = eval_subvolume_data.fullfnames
    eval_labels = sorted(label_data.name for label_data in eval_labels_data)
    if train_labels != eval_labels:
        raise ValueError(
            'Train labels and eval labels are not the same '
            '(train=[{}], eval=[{}]).'.format(train_labels, eval_labels)
            )
    labels = train_labels
    eval_subvolume_slice_labels = volumes.load_labels(eval_labels_data)
    validations.validate_annotation_data(full_volume, eval_subvolume_data, eval_labels_data)
    for (i, fullfname) in enumerate(eval_subvolume_fullfnames):
        labels_in_slice = {
            label for label in np.unique(
                eval_subvolume_slice_labels[i*slice_size:(i+1)*slice_size]
                ).tolist()
            if label < volumes.FIRST_CONTROL_LABEL
            }
        if len(labels_in_slice) != len(labels):
            raise ValueError(
                'Eval labels of slice {} do not cover all labels '
                '(missing labels = [{}]).'.format(fullfname, set(labels) - labels_in_slice)
                )

    listener.log_output('> Config file')
    if isinstance(config, str):
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    validations.validate_json_with_schema_file(config_data, 'tune.json')
    segmenter = segmenters.Segmenter(
        labels,
        full_volume,
        {
            'featuriser': config_data['featuriser'],
            'classifier': config_data['classifier'],
            'training_set': config_data['training_set'],
            },
        allow_random=True
        )
    
    listener.log_output('> Result file')
    if results_fullfname is not None:
        validations.check_filename(results_fullfname, '.txt', False)
    tuning_results_file = results.TuningResultsFile(results_fullfname)

    listener.log_output('> Checkpoint')
    if checkpoint_fullfname is not None:
        validations.check_filename(checkpoint_fullfname, '.json', False)
    checkpoint = checkpoints.CheckpointManager(
        'tune',
        checkpoint_fullfname,
        restart_checkpoint
        )

    listener.log_output('> Initialising')
    hash_function.init(slice_shape, seed=0)
    training_set = trainingsets.TrainingSet(None)
    
    return (config_data, full_volume, slice_shape, slice_size, segmenter, train_subvolume_fullfnames, train_subvolume_slice_labels, eval_subvolume_fullfnames, eval_subvolume_slice_labels, training_set, hash_function, tuning_results_file, checkpoint)
    
    
#########################################
def _hashing_train_subvolume_slices(
        full_volume, train_subvolume_fullfnames, hash_function, listener
    ):
    '''Hashing train subvolume slices stage.'''
    train_subvolume_hashes = np.empty(
        (len(train_subvolume_fullfnames), hash_function.hash_size),
        full_volume.get_hashes_dtype())
    for (i, fullfname) in enumerate(train_subvolume_fullfnames):
        img_data = images.load_image(fullfname)
        train_subvolume_hashes[i, :] = hash_function.apply(img_data)
    volume_slice_indexes_in_train_subvolume = volumes.get_volume_slice_indexes_in_subvolume(
        full_volume.get_hashes_array()[:], train_subvolume_hashes  #Load the hashes eagerly.
        )
    listener.log_output('> Train subvolume to volume file name mapping found:')
    for (subvolume_index, volume_index) in enumerate(
            volume_slice_indexes_in_train_subvolume
        ):
        listener.log_output('>> {} -> volume slice #{}'.format(
            train_subvolume_fullfnames[subvolume_index], volume_index+1
            ))
    
    return (volume_slice_indexes_in_train_subvolume,)


#########################################
def _hashing_eval_subvolume_slices(
        full_volume, eval_subvolume_fullfnames, hash_function, listener
    ):
    '''Hashing eval subvolume slices stage.'''
    eval_subvolume_hashes = np.empty(
        (len(eval_subvolume_fullfnames), hash_function.hash_size),
        full_volume.get_hashes_dtype())
    for (i, fullfname) in enumerate(eval_subvolume_fullfnames):
        img_data = images.load_image(fullfname)
        eval_subvolume_hashes[i, :] = hash_function.apply(img_data)
    volume_slice_indexes_in_eval_subvolume = volumes.get_volume_slice_indexes_in_subvolume(
        full_volume.get_hashes_array()[:], eval_subvolume_hashes  #Load the hashes eagerly.
        )
    listener.log_output('> Eval subvolume to volume file name mapping found:')
    for (subvolume_index, volume_index) in enumerate(
            volume_slice_indexes_in_eval_subvolume
        ):
        listener.log_output('>> {} -> volume slice #{}'.format(
            eval_subvolume_fullfnames[subvolume_index], volume_index+1
            ))
    
    return (volume_slice_indexes_in_eval_subvolume,)


#########################################
def _tuning(
        config_data, segmenter, slice_shape, slice_size, full_volume, train_subvolume_fullfnames, train_subvolume_slice_labels, volume_slice_indexes_in_train_subvolume, eval_subvolume_fullfnames, eval_subvolume_slice_labels, volume_slice_indexes_in_eval_subvolume, training_set, tuning_results_file, checkpoint, max_processes, max_batch_memory, listener
    ):
    '''Tuning stage.'''
    parameters_visited = set()
    with checkpoint.apply('create_results_file') as skip:
        if skip is not None:
            listener.log_output('> Continuing use of existing results file')
            raise skip
        tuning_results_file.create(segmenter.classifier.labels)
    with checkpoint.apply('tune') as skip:
        if skip is not None:
            raise skip
        start = checkpoint.get_next_to_process('tune_prog')
        for iteration in range(1, config_data['tuning']['num_iterations'] + 1):
            while True:
                segmenter.regenerate()
                params = segmenter.featuriser.get_params()
                if params not in parameters_visited:
                    parameters_visited.add(params)
                    break
            if iteration < start:
                continue
            with checkpoint.apply('tune_prog'):
                if iteration > 1:
                    listener.log_output('-')
                with times.Timer() as sub_timer:
                    listener.log_output('> Iteration {}'.format(iteration))
                    listener.log_output('>> {}'.format(json.dumps(segmenter.get_config())))
                    
                    training_set.create(slice_size*len(train_subvolume_fullfnames), segmenter.featuriser.get_feature_size())
                
                    training_set.get_labels_array()[:] = train_subvolume_slice_labels

                    best_block_shape = arrayprocs.get_optimal_block_size(
                        slice_shape,
                        full_volume.get_dtype(),
                        segmenter.featuriser.get_context_needed(),
                        max_processes,
                        max_batch_memory,
                        implicit_depth=True
                        )
                    for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_train_subvolume):
                        segmenter.featuriser.featurise(
                            full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                            slice_index=volume_slice_index,
                            block_rows=best_block_shape[0],
                            block_cols=best_block_shape[0],
                            output=training_set.get_features_array(),
                            output_start_row_index=i*slice_size,
                            n_jobs=max_processes
                            )
                    
                    segmenter.train(training_set, max_processes)
                    
                    total_ious = [0 for _ in range(len(segmenter.classifier.labels))]
                    for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_eval_subvolume):
                        slice_features = segmenter.featuriser.featurise(
                            full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                            slice_index=volume_slice_index,
                            block_rows=best_block_shape[0],
                            block_cols=best_block_shape[0],
                            n_jobs=max_processes
                            )

                        prediction = segmenter.segment_to_label_indexes(slice_features, max_processes)
                        prediction = prediction.reshape(slice_shape)

                        slice_labels = eval_subvolume_slice_labels[i*slice_size:(i+1)*slice_size]
                        slice_labels = slice_labels.reshape(slice_shape)

                        ious = evaluations.get_intersection_over_union(
                            prediction, slice_labels, len(segmenter.classifier.labels)
                            )
                        for label_index in range(len(segmenter.classifier.labels)):
                            total_ious[label_index] += ious[label_index]
                    
                    listener.log_output('>> Results:')
                    average_ious = [total_iou/len(eval_subvolume_fullfnames) for total_iou in total_ious]
                    for (label, iou) in zip(segmenter.classifier.labels, average_ious):
                        listener.log_output('>>> {}: {:.3%}'.format(label, iou))
                    tuning_results_file.append(segmenter.get_config(), ious)
                listener.log_output('> Duration: {}'.format(times.get_readable_duration(sub_timer.duration)))
    
    return ()

#########################################
def main(
        preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
        eval_subvolume_dir, eval_label_dirs, config,
        results_fullfname, checkpoint_fullfname, restart_checkpoint,
        max_processes, max_batch_memory, listener=ProgressListener()
    ):
    '''
    Find Train a classifier model to segment volumes based on manually labelled slices.

    :param str preproc_volume_fullfname: The full file name (with path) to the preprocessed
        volume HDF file.
    :param str train_subvolume_dir: The path to the directory containing copies from the full
        volume slices that were labelled for training.
    :param list train_label_dirs: A list of paths to the directories containing labelled
        slices for training with the number of labels being equal to the number of directories and
        the number of images in each directory being equal to the number of train subvolume
        images.
    :param str eval_subvolume_dir: The path to the directory containing copies from the full
        volume slices that were labelled for evaluation.
    :param list eval_label_dirs: A list of paths to the directories containing labelled
        slices for evaluation with the number of labels being equal to the number of directories
        and the number of images in each directory being equal to the number of eval subvolume
        images.
    :param config: The configuration to use when tuning (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the eval configuration.
    :type config: str or dict
    :param results_fullfname: Full file name (with path) to the text file to create. If None
        then results will be returned instead of saved.
    :type results_fullfname: str or None
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
            listener.overall_progress_start(4)

            listener.log_output('Starting tuning process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (config_data, full_volume, slice_shape, slice_size, segmenter, train_subvolume_fullfnames, train_subvolume_slice_labels, eval_subvolume_fullfnames, eval_subvolume_slice_labels, training_set, hash_function, tuning_results_file, checkpoint) = _loading_data(
                    preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
                    eval_subvolume_dir, eval_label_dirs, config,
                    results_fullfname, checkpoint_fullfname, restart_checkpoint,
                    listener
                    )
            listener.log_output('Input data')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Hashing train subvolume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing train subvolume slices')
            with times.Timer() as timer:
                (volume_slice_indexes_in_train_subvolume,) = _hashing_train_subvolume_slices(full_volume, train_subvolume_fullfnames, hash_function, listener)
            listener.log_output('Slices hashed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################
            
            listener.overall_progress_update(3, 'Hashing eval subvolume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing eval subvolume slices')
            with times.Timer() as timer:
                (volume_slice_indexes_in_eval_subvolume,) = _hashing_eval_subvolume_slices(full_volume, eval_subvolume_fullfnames, hash_function, listener)
            listener.log_output('Slices hashed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')
            
            ###################
            
            listener.overall_progress_update(4, 'Tuning')
            listener.log_output(times.get_timestamp())
            listener.log_output('Tuning')
            with times.Timer() as timer:
                () = _tuning(config_data, segmenter, slice_shape, slice_size, full_volume, train_subvolume_fullfnames, train_subvolume_slice_labels, volume_slice_indexes_in_train_subvolume, eval_subvolume_fullfnames, eval_subvolume_slice_labels, volume_slice_indexes_in_eval_subvolume, training_set, tuning_results_file, checkpoint, max_processes, max_batch_memory, listener)
            listener.log_output('Tuned')
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