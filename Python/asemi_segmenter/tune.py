'''Tune command.'''

import json
import memory_profiler
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
from asemi_segmenter.lib import datasets
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
        eval_subvolume_dir, eval_label_dirs, config,
        results_fullfname, checkpoint_fullfname, checkpoint_init,
        max_processes, max_batch_memory, listener
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
    
    listener.log_output('> Train subvolume')
    listener.log_output('>> {}'.format(train_subvolume_dir))
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
    validations.validate_annotation_data(full_volume, train_subvolume_data, train_labels_data)

    listener.log_output('> Evaluation subvolume')
    listener.log_output('>> {}'.format(eval_subvolume_dir))
    eval_subvolume_data = volumes.load_volume_dir(eval_subvolume_dir)
    eval_subvolume_fullfnames = eval_subvolume_data.fullfnames

    listener.log_output('> Evaluation labels')
    eval_labels_data = []
    for label_dir in eval_label_dirs:
        listener.log_output('>> {}'.format(label_dir))
        label_data = volumes.load_label_dir(label_dir)
        eval_labels_data.append(label_data)
        listener.log_output('>>> {}'.format(label_data.name))
    eval_labels = sorted(label_data.name for label_data in eval_labels_data)
    if train_labels != eval_labels:
        raise ValueError(
            'Train labels and evaluation labels are not the same '
            '(train=[{}], eval=[{}]).'.format(train_labels, eval_labels)
            )
    labels = train_labels
    validations.validate_annotation_data(full_volume, eval_subvolume_data, eval_labels_data)

    listener.log_output('> Config')
    if isinstance(config, str):
        listener.log_output('>> {}'.format(config))
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    validations.validate_json_with_schema_file(config_data, 'tune.json')
    if config_data['evaluation_set']['sample_size_per_label'] == 0:
        raise ValueError('Evaluation set configuration is invalid as sample_size_per_label cannot be 0.')
    if config_data['evaluation_set']['same_as_train'] and config_data['training_set']['sample_size_per_label'] == -1:
        raise ValueError('Evaluation set configuration is invalid as same_as_train can only be true if training_set sample_size_per_label is not -1.')
    
    listener.log_output('> Result')
    if results_fullfname is not None:
        listener.log_output('>> {}'.format(results_fullfname))
        validations.check_filename(results_fullfname, '.txt', False)
    evaluation = evaluations.IntersectionOverUnionEvaluation(len(labels))
    tuning_results_file = results.TuningResultsFile(results_fullfname, evaluation)

    listener.log_output('> Checkpoint')
    if checkpoint_fullfname is not None:
        listener.log_output('>> {}'.format(checkpoint_fullfname))
        validations.check_filename(checkpoint_fullfname, '.json', False)
    checkpoint = checkpoints.CheckpointManager(
        'tune',
        checkpoint_fullfname,
        initial_content=checkpoint_init
        )

    listener.log_output('> Initialising')
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
    hash_function.init(slice_shape, seed=0)
    training_set = datasets.DataSet(None)
    
    listener.log_output('> Other parameters:')
    listener.log_output('>> max_processes: {}'.format(max_processes))
    listener.log_output('>> max_batch_memory: {}GB'.format(max_batch_memory))
    
    return (config_data, full_volume, slice_shape, slice_size, segmenter, train_subvolume_fullfnames, train_labels_data, eval_subvolume_fullfnames, eval_labels_data, training_set, hash_function, evaluation, tuning_results_file, checkpoint)
    
    
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
    listener.log_output('> Evaluation subvolume to volume file name mapping found:')
    for (subvolume_index, volume_index) in enumerate(
            volume_slice_indexes_in_eval_subvolume
        ):
        listener.log_output('>> {} -> volume slice #{}'.format(
            eval_subvolume_fullfnames[subvolume_index], volume_index+1
            ))
    
    return (volume_slice_indexes_in_eval_subvolume,)


#########################################
def _constructing_labels_dataset(
        train_labels_data, eval_labels_data
    ):
    '''Constructing labels dataset stage.'''
    train_subvolume_slice_labels = volumes.load_labels(train_labels_data)
    eval_subvolume_slice_labels = volumes.load_labels(eval_labels_data)
    
    return (train_subvolume_slice_labels, eval_subvolume_slice_labels)


#########################################
def _tuning(
        config_data, segmenter, slice_shape, slice_size, full_volume, train_subvolume_slice_labels, volume_slice_indexes_in_train_subvolume, eval_subvolume_slice_labels, volume_slice_indexes_in_eval_subvolume, training_set, evaluation, tuning_results_file, checkpoint, max_processes, max_batch_memory, listener, extra_col_names, extra_col_values
    ):
    '''Tuning stage.'''
    train_sample_size_per_label = config_data['training_set']['sample_size_per_label']
    eval_sample_size_per_label = config_data['evaluation_set']['sample_size_per_label']
    eval_same_as_train = config_data['evaluation_set']['same_as_train']
    
    listener.log_output('> Train label sizes:')
    if train_sample_size_per_label != -1:
        (train_voxel_indexes, train_label_positions) = datasets.sample_voxels(
            train_subvolume_slice_labels,
            train_sample_size_per_label,
            len(segmenter.classifier.labels),
            volume_slice_indexes_in_train_subvolume,
            slice_shape,
            seed=0
            )
        for (label, label_slice) in zip(segmenter.classifier.labels, train_label_positions):
            listener.log_output('>> {}: {}'.format(label, label_slice.stop - label_slice.start))
    else:
        for (label_index, label) in enumerate(segmenter.classifier.labels):
            listener.log_output('>> {}: {}'.format(label, np.sum(train_subvolume_slice_labels == label_index)))
    
    listener.log_output('> Evaluation label sizes:')
    if eval_sample_size_per_label != -1:
        (eval_voxel_indexes, eval_label_positions) = datasets.sample_voxels(
            eval_subvolume_slice_labels,
            eval_sample_size_per_label,
            len(segmenter.classifier.labels),
            volume_slice_indexes_in_eval_subvolume,
            slice_shape,
            skip=0 if not eval_same_as_train else train_sample_size_per_label,
            seed=0
            )
        for (label, label_slice) in zip(segmenter.classifier.labels, eval_label_positions):
            listener.log_output('>> {}: {}'.format(label, label_slice.stop - label_slice.start))
    else:
        for (label_index, label) in enumerate(segmenter.classifier.labels):
            listener.log_output('>> {}: {}'.format(label, np.sum(eval_subvolume_slice_labels == label_index)))
        
    parameters_visited = set()
    with checkpoint.apply('create_results_file') as skip:
        if skip is not None:
            listener.log_output('> Continuing use of checkpointed results file')
            raise skip
        tuning_results_file.create(segmenter.classifier.labels, extra_col_names)
    with checkpoint.apply('tune') as skip:
        if skip is not None:
            raise skip
        start = checkpoint.get_next_to_process('tune_prog')
        listener.current_progress_start(start, config_data['tuning']['num_iterations'])
        for iteration in range(1, config_data['tuning']['num_iterations'] + 1):
            evaluation.reset()
            while True:
                segmenter.regenerate()
                params = segmenter.get_params()
                if params not in parameters_visited:
                    parameters_visited.add(params)
                    break
            if iteration - 1 < start:
                continue
            with checkpoint.apply('tune_prog'):
                with times.Timer() as sub_timer:
                    best_block_shape = arrayprocs.get_optimal_block_size(
                        slice_shape,
                        full_volume.get_dtype(),
                        segmenter.featuriser.get_context_needed(),
                        max_processes,
                        max_batch_memory,
                        implicit_depth=True
                        )
                    
                    if train_sample_size_per_label != -1:
                        training_set.create(
                            len(train_voxel_indexes),
                            segmenter.featuriser.get_feature_size()
                            )
                        for (label_index, label_position) in enumerate(train_label_positions):
                            training_set.get_labels_array()[label_position] = label_index
                        segmenter.featuriser.featurise_voxels(
                            full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                            train_voxel_indexes,
                            output=training_set.get_features_array(),
                            n_jobs=max_processes
                            )
                    else:
                        training_set.create(
                            slice_size*len(volume_slice_indexes_in_train_subvolume),
                            segmenter.featuriser.get_feature_size()
                            )
                        training_set.get_labels_array()[:] = train_subvolume_slice_labels
                        for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_train_subvolume):
                            segmenter.featuriser.featurise_slice(
                                full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                slice_index=volume_slice_index,
                                block_rows=best_block_shape[0],
                                block_cols=best_block_shape[1],
                                output=training_set.get_features_array(),
                                output_start_row_index=i*slice_size,
                                n_jobs=max_processes
                                )
                        training_set = training_set.without_control_labels()
                    
                    def memory_scope(result):
                        segmenter.train(training_set, max_processes)
                        
                        iou_lists = [[] for _ in range(len(segmenter.classifier.labels))]
                        if eval_sample_size_per_label != -1:
                            eval_set = datasets.DataSet(None)
                            eval_set.create(
                                len(eval_voxel_indexes),
                                segmenter.featuriser.get_feature_size()
                                )
                            for (label_index, label_position) in enumerate(eval_label_positions):
                                eval_set.get_labels_array()[label_position] = label_index
                            
                            with times.Timer() as featuriser_timer:
                                segmenter.featuriser.featurise_voxels(
                                    full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                    eval_voxel_indexes,
                                    output=eval_set.get_features_array(),
                                    n_jobs=max_processes
                                    )
                            
                            with times.Timer() as classifier_timer:
                                prediction = segmenter.segment_to_label_indexes(eval_set.get_features_array(), max_processes)
                        
                            evaluation.evaluate(prediction, eval_set.get_labels_array())
                        else:
                            for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_eval_subvolume):
                                with times.Timer() as featuriser_timer:
                                    slice_features = segmenter.featuriser.featurise_slice(
                                        full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                        slice_index=volume_slice_index,
                                        block_rows=best_block_shape[0],
                                        block_cols=best_block_shape[1],
                                        n_jobs=max_processes
                                        )
                                
                                with times.Timer() as classifier_timer:
                                    prediction = segmenter.segment_to_label_indexes(slice_features, max_processes)
                            
                                evaluation.evaluate(prediction, eval_subvolume_slice_labels)
                            
                        result['featuriser_time'] = featuriser_timer.duration
                        result['classifier_time'] = classifier_timer.duration
                    result = dict()
                    max_memory_mb = max(memory_profiler.memory_usage((memory_scope, (result,)), interval=0))
                    
                    ious = evaluation.get_global_result_per_label()
                    global_iou = evaluation.get_global_result()
                    tuning_results_file.add(
                        segmenter.get_config(),
                        result['featuriser_time'],
                        result['classifier_time'],
                        max_memory_mb,
                        extra_col_values
                        )
            listener.current_progress_update(iteration)
        listener.current_progress_end()
    
    return ()

#########################################
def main(
        preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
        eval_subvolume_dir, eval_label_dirs, config,
        results_fullfname, checkpoint_fullfname, checkpoint_init,
        max_processes, max_batch_memory, listener=ProgressListener(),
        debug_mode=False, extra_result_col_names=[], extra_result_col_values=[]
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
    :param dict checkpoint_init: The checkpoint data to initialise the checkpoint with,
        including the checkpoint file (only data about this particular command will be
        overwritten). If None then checkpoint is checkpoint file content if file exists,
        otherwise the checkpoint will be empty. To restart checkpoint set to empty dictionary.
    :param int max_processes: The maximum number of processes to use concurrently.
    :param float max_batch_memory: The maximum number of gigabytes to use between all processes.
    :param ProgressListener listener: The command's progress listener.
    :param bool debug_mode: Whether to show full error messages or just simple ones.
    :param list extra_result_col_names: Names of any extra columns to add to the result file.
    :param list extra_result_col_values: Values (fixed) of any extra columns to add to the result file.
    '''
    full_volume = None
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(5)

            listener.log_output('Starting tuning process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (config_data, full_volume, slice_shape, slice_size, segmenter, train_subvolume_fullfnames, train_labels_data, eval_subvolume_fullfnames, eval_labels_data, training_set, hash_function, evaluation, tuning_results_file, checkpoint) = _loading_data(
                    preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
                    eval_subvolume_dir, eval_label_dirs, config,
                    results_fullfname, checkpoint_fullfname, checkpoint_init, max_processes, max_batch_memory, listener
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
            
            listener.overall_progress_update(3, 'Hashing Evaluation subvolume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing evaluation subvolume slices')
            with times.Timer() as timer:
                (volume_slice_indexes_in_eval_subvolume,) = _hashing_eval_subvolume_slices(full_volume, eval_subvolume_fullfnames, hash_function, listener)
            listener.log_output('Slices hashed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')
            
            ###################

            listener.overall_progress_update(4, 'Constructing labels dataset')
            listener.log_output(times.get_timestamp())
            listener.log_output('Constructing labels dataset')
            with times.Timer() as timer:
                (train_subvolume_slice_labels, eval_subvolume_slice_labels) = _constructing_labels_dataset(train_labels_data, eval_labels_data)
            listener.log_output('Labels dataset constructed')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')
            
            ###################
            
            listener.overall_progress_update(5, 'Tuning')
            listener.log_output(times.get_timestamp())
            listener.log_output('Tuning')
            with times.Timer() as timer:
                () = _tuning(config_data, segmenter, slice_shape, slice_size, full_volume, train_subvolume_slice_labels, volume_slice_indexes_in_train_subvolume, eval_subvolume_slice_labels, volume_slice_indexes_in_eval_subvolume, training_set, evaluation, tuning_results_file, checkpoint, max_processes, max_batch_memory, listener, extra_result_col_names, extra_result_col_values)
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
        if debug_mode:
            raise
        else:
            listener.error_output(str(ex))
    finally:
        if full_volume is not None:
            full_volume.close()