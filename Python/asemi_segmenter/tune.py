'''Tune command.'''

import sys
import random
import json
import memory_profiler
import numpy as np
from asemi_segmenter import listeners
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import evaluations
from asemi_segmenter.lib import hashfunctions
from asemi_segmenter.lib import images
from asemi_segmenter.lib import results
from asemi_segmenter.lib import segmenters
from asemi_segmenter.lib import times
from asemi_segmenter.lib import datasets
from asemi_segmenter.lib import samplers
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes
from asemi_segmenter.lib import featurisers


#########################################
def _loading_data(
        preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
        eval_subvolume_dir, eval_label_dirs, config, search_results_fullfname,
        best_result_fullfname, parameter_selection_timeout, use_features_table,
        features_table_fullfname, train_sample_seed, eval_sample_seed,
        parameter_selection_seed, checkpoint_fullfname, checkpoint_namespace,
        reset_checkpoint, checkpoint_init, max_processes, max_batch_memory, use_gpu, listener
    ):
    '''Loading data stage.'''
    if train_sample_seed is None:
        train_sample_seed = random.randrange(sys.maxsize)
    if eval_sample_seed is None:
        eval_sample_seed = random.randrange(sys.maxsize)
    if parameter_selection_seed is None:
        parameter_selection_seed = random.randrange(sys.maxsize)

    listener.log_output('> Volume')
    listener.log_output('>> {}'.format(preproc_volume_fullfname))
    validations.check_filename(preproc_volume_fullfname, '.hdf', True)
    full_volume = volumes.FullVolume(preproc_volume_fullfname)
    full_volume.load(as_readonly=True)
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
    if 'samples_to_skip_per_label' not in config_data['training_set']:
        config_data['training_set']['samples_to_skip_per_label'] = 0
    if 'samples_to_skip_per_label' not in config_data['evaluation_set']:
        config_data['evaluation_set']['samples_to_skip_per_label'] = 0
    sampler_factory = samplers.SamplerFactory(seed=parameter_selection_seed)
    for variable_name in config_data['variables']:
        if config_data['variables'][variable_name]['type'] == 'integer':
            sampler_factory.create_integer_sampler(
                config_data['variables'][variable_name]['min'],
                config_data['variables'][variable_name]['max'],
                config_data['variables'][variable_name]['distribution'],
                name=variable_name
                )
        elif config_data['variables'][variable_name]['type'] == 'float':
            sampler_factory.create_float_sampler(
                config_data['variables'][variable_name]['min'],
                config_data['variables'][variable_name]['max'],
                config_data['variables'][variable_name]['distribution'],
                name=variable_name
                )
    segmenter = segmenters.Segmenter(
        labels,
        full_volume,
        {
            'featuriser': config_data['featuriser'],
            'classifier': config_data['classifier'],
            'training_set': config_data['training_set']
            },
        sampler_factory=sampler_factory,
        use_gpu=use_gpu
        )
    listener.log_output('>> Search space size: {}'.format(sampler_factory.get_sample_space_size()))

    listener.log_output('> Search results')
    if search_results_fullfname is not None:
        listener.log_output('>> {}'.format(search_results_fullfname))
        validations.check_filename(search_results_fullfname, '.txt', False)
    evaluation = evaluations.IntersectionOverUnionEvaluation(len(labels))
    tuning_results_file = results.TuningResultsFile(search_results_fullfname, evaluation)

    listener.log_output('> Best result')
    if best_result_fullfname is not None:
        listener.log_output('>> {}'.format(best_result_fullfname))
        validations.check_filename(best_result_fullfname, '.json', False)

    listener.log_output('> Parameter selection timeout')
    listener.log_output('>> {}'.format(parameter_selection_timeout))
    if parameter_selection_timeout <= 0:
        raise ValueError('Must be a positive number.')

    listener.log_output('> Feature table')
    if use_features_table:
        if features_table_fullfname is not None:
            listener.log_output('>> {}'.format(features_table_fullfname))
            validations.check_filename(features_table_fullfname, '.hdf', False)
        else:
            listener.log_output('>> in memory')
        features_table = featurisers.FeaturesTable(features_table_fullfname)
    else:
        features_table = None

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
    training_set = datasets.DataSet(None)


    listener.log_output('> Other parameters:')
    listener.log_output('>> train sample seed: {}'.format(train_sample_seed))
    listener.log_output('>> evaluation sample seed: {}'.format(eval_sample_seed))
    listener.log_output('>> parameter selection seed: {}'.format(parameter_selection_seed))
    listener.log_output('>> reset_checkpoint: {}'.format(reset_checkpoint))
    listener.log_output('>> max_processes: {}'.format(max_processes))
    listener.log_output('>> max_batch_memory: {}GB'.format(max_batch_memory))

    return (config_data, full_volume, slice_shape, slice_size, segmenter, train_subvolume_fullfnames, train_labels_data, eval_subvolume_fullfnames, eval_labels_data, training_set, hash_function, evaluation, tuning_results_file, features_table, train_sample_seed, eval_sample_seed, parameter_selection_seed, checkpoint)


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
        config_data, segmenter, slice_shape, slice_size, full_volume, train_subvolume_slice_labels, volume_slice_indexes_in_train_subvolume, eval_subvolume_slice_labels, volume_slice_indexes_in_eval_subvolume, training_set, evaluation, parameter_selection_timeout, tuning_results_file, features_table, train_sample_seed, eval_sample_seed, checkpoint, max_processes, max_batch_memory, listener, extra_col_names, extra_col_values
    ):
    '''Tuning stage.'''
    train_sample_size_per_label = config_data['training_set']['sample_size_per_label']
    eval_sample_size_per_label = config_data['evaluation_set']['sample_size_per_label']

    listener.log_output('> Train label sizes:')
    if train_sample_size_per_label != -1:
        (train_voxel_indexes, train_label_positions) = datasets.sample_voxels(
            train_subvolume_slice_labels,
            train_sample_size_per_label,
            len(segmenter.classifier.labels),
            volume_slice_indexes_in_train_subvolume,
            slice_shape,
            config_data['training_set']['samples_to_skip_per_label'],
            seed=train_sample_seed
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
            config_data['evaluation_set']['samples_to_skip_per_label'],
            seed=eval_sample_seed
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

    tuning_results_file.load()

    if features_table is not None:
        with checkpoint.apply('create_features_table') as skip:
            if skip is not None:
                listener.log_output('> Continuing use of checkpointed features table file')
                raise skip
            features_table.create()

        features_table.load()

    last_global_iteration = 0

    class TimedOut(Exception):
        '''Time out exception.'''
        pass

    listener.log_output('> Running global search')
    with checkpoint.apply('global_tune') as skip:
        if skip is not None:
            raise skip
        start = checkpoint.get_next_to_process('global_tune_prog')
        try:
            listener.current_progress_start(start, config_data['tuning']['num_global_iterations'])
            for iteration in range(1, config_data['tuning']['num_global_iterations'] + 1):
                evaluation.reset()
                with times.Timer() as timeout_timer:
                    while True:
                        segmenter.sampler_factory.resample_all()
                        segmenter.refresh_params()
                        params = segmenter.get_params()
                        if params not in parameters_visited:
                            parameters_visited.add(params)
                            break
                        if timeout_timer.get_current_duration() > parameter_selection_timeout:
                            raise TimedOut()
                if iteration - 1 < start:
                    continue
                with checkpoint.apply('global_tune_prog'):
                    with times.Timer() as sub_timer:
                        best_block_shape = arrayprocs.get_optimal_block_size(
                            slice_shape,
                            full_volume.get_dtype(),
                            segmenter.featuriser.get_context_needed(),
                            max_processes,
                            max_batch_memory,
                            num_implicit_slices=1,
                            feature_size=segmenter.featuriser.get_feature_size(),
                            feature_dtype=featurisers.feature_dtype
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
                                dataset_name='training_set',
                                features_table=features_table
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
                                    slice_range=slice(volume_slice_index, volume_slice_index+1),
                                    block_shape=best_block_shape,
                                    output=training_set.get_features_array(),
                                    output_start_row_index=i*slice_size,
                                    n_jobs=max_processes
                                    )
                            training_set = training_set.without_control_labels()

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

                            segmenter.featuriser.featurise_voxels(
                                full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                eval_voxel_indexes,
                                output=eval_set.get_features_array(),
                                dataset_name='evaluation_set',
                                features_table=features_table
                                )

                            prediction = segmenter.segment_to_label_indexes(eval_set.get_features_array(), max_processes)

                            evaluation.evaluate(prediction, eval_set.get_labels_array())
                        else:
                            for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_eval_subvolume):
                                slice_features = segmenter.featuriser.featurise_slice(
                                    full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                    slice_range=slice(volume_slice_index, volume_slice_index+1),
                                    block_shape=best_block_shape,
                                    n_jobs=max_processes
                                    )

                                prediction = segmenter.segment_to_label_indexes(slice_features, max_processes)

                                evaluation.evaluate(prediction, eval_subvolume_slice_labels[i*slice_size:(i+1)*slice_size])

                    tuning_results_file.add(
                        'global',
                        iteration,
                        segmenter.get_config(),
                        sub_timer.duration,
                        extra_col_values
                        )
                    last_global_iteration = iteration
                listener.current_progress_update(iteration)
            listener.current_progress_end()
        except TimedOut as ex:
            listener.current_progress_end()
            listener.log_output('>> Next parameter selection process timed out')

    listener.log_output('> Running local search')
    with checkpoint.apply('local_tune') as skip:
        if skip is not None:
            raise skip
        start = checkpoint.get_next_to_process('local_tune_prog')
        try:
            listener.current_progress_start(start, config_data['tuning']['num_local_iterations'])
            for iteration in range(1, config_data['tuning']['num_local_iterations'] + 1):
                evaluation.reset()
                with times.Timer() as timeout_timer:
                    while True:
                        segmenter.set_sampler_values(tuning_results_file.best_config)
                        segmenter.sampler_factory.resample_random_one()
                        segmenter.refresh_params()
                        params = segmenter.get_params()
                        if params not in parameters_visited:
                            parameters_visited.add(params)
                            break
                        if timeout_timer.get_current_duration() > parameter_selection_timeout:
                            raise TimedOut()
                if iteration - 1 < start:
                    continue
                with checkpoint.apply('local_tune_prog'):
                    with times.Timer() as sub_timer:
                        best_block_shape = arrayprocs.get_optimal_block_size(
                            slice_shape,
                            full_volume.get_dtype(),
                            segmenter.featuriser.get_context_needed(),
                            max_processes,
                            max_batch_memory,
                            num_implicit_slices=1,
                            feature_size=segmenter.featuriser.get_feature_size(),
                            feature_dtype=featurisers.feature_dtype
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
                                dataset_name='training_set',
                                features_table=features_table
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
                                    slice_range=slice(volume_slice_index, volume_slice_index+1),
                                    block_shape=best_block_shape,
                                    output=training_set.get_features_array(),
                                    output_start_row_index=i*slice_size,
                                    n_jobs=max_processes
                                    )
                            training_set = training_set.without_control_labels()

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

                            segmenter.featuriser.featurise_voxels(
                                full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                eval_voxel_indexes,
                                output=eval_set.get_features_array(),
                                dataset_name='evaluation_set',
                                features_table=features_table
                                )

                            prediction = segmenter.segment_to_label_indexes(eval_set.get_features_array(), max_processes)

                            evaluation.evaluate(prediction, eval_set.get_labels_array())
                        else:
                            for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_eval_subvolume):
                                slice_features = segmenter.featuriser.featurise_slice(
                                    full_volume.get_scale_arrays(segmenter.featuriser.get_scales_needed()),
                                    slice_range=slice(volume_slice_index, volume_slice_index+1),
                                    block_shape=best_block_shape,
                                    n_jobs=max_processes
                                    )

                                prediction = segmenter.segment_to_label_indexes(slice_features, max_processes)

                                evaluation.evaluate(prediction, eval_subvolume_slice_labels[i*slice_size:(i+1)*slice_size])

                    tuning_results_file.add(
                        'local',
                        last_global_iteration + iteration,
                        segmenter.get_config(),
                        sub_timer.duration,
                        extra_col_values
                        )
                listener.current_progress_update(iteration)
            listener.current_progress_end()
        except TimedOut as ex:
            listener.current_progress_end()
            listener.log_output('>> Next parameter selection process timed out')

    best_config = dict()
    best_config['featuriser'] = tuning_results_file.best_config['featuriser']
    best_config['classifier'] = tuning_results_file.best_config['classifier']
    best_config['training_set'] = config_data['output']['training_set']

    return (best_config,)

#########################################
def _saving_best_config(best_result_fullfname, best_config, listener):
    '''Saving best config stage.'''
    if best_result_fullfname is not None:
        with open(best_result_fullfname, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent='\t')
    else:
        listener.log_output('Config not to be saved')

    return ()

#########################################
def main(
        preproc_volume_fullfname,
        train_subvolume_dir,
        train_label_dirs,
        eval_subvolume_dir,
        eval_label_dirs,
        config,
        search_results_fullfname,
        best_result_fullfname=None,
        parameter_selection_timeout=1,
        use_features_table=False,
        features_table_fullfname=None,
        extra_result_col_names=[],
        extra_result_col_values=[],
        train_sample_seed=None,
        eval_sample_seed=None,
        parameter_selection_seed=None,
        checkpoint_fullfname=None,
        checkpoint_namespace='tune',
        reset_checkpoint=False,
        checkpoint_init=dict(),
        max_processes=-1,
        max_batch_memory=1,
        use_gpu=False,
        listener=listeners.ProgressListener(),
        debug_mode=False
    ):
    '''
    Find the best parameters for a segmenter based on manually labelled slices.

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
    :param str search_results_fullfname: Full file name (with path) to the text file that will
        contain all the configurations tested. If None then no file will be saved.
    :param str best_result_fullfname: Full file name (with path) to the JSON file that will
        contain the best configuration found as a JSON encoded configuration file. If None
        then no file will be saved.
    :param int parameter_selection_timeout: The next set of parameters to try are randomly
        generated until a new set is found. This is the number of seconds to allow the command
        to randomly generate parameters before it times out and ends the search.
    :param bool use_features_table: Whether to create a lookup table speeding up tuning or not.
    :param str features_table_fullfname: Full file name (with path) to the HDF file that will
        contain precomputed features to speed up the search. If None then table will be kept in
        memory instead. Used on both training and evaluation sets but only if they are sampled
        (sample_size_per_label is not -1).
    :param list extra_result_col_names: Names of any extra columns to add to the result file.
    :param list extra_result_col_values: Values (fixed) of any extra columns to add to the result file.
    :param int train_sample_seed: Seed for the random number generator which samples training
        voxels. If None then a seed will be generated randomly.
    :param int eval_sample_seed: Seed for the random number generator which samples evaluation
        voxels. If None then the random number generator will be non-deterministic.
    :param int parameter_selection_seed: Seed for the random number generator which samples
        parameters. If None then the random number generator will be non-deterministic.
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
    :param bool use_gpu: Whether to use the GPU for computing features.
    :param ProgressListener listener: The command's progress listener.
    :param bool debug_mode: Whether to show full error messages or just simple ones.
    :return: The config data of the best segmenter found.
    :rtype: dict
    '''
    full_volume = None
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(6)

            listener.log_output('Starting tuning process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (config_data, full_volume, slice_shape, slice_size, segmenter, train_subvolume_fullfnames, train_labels_data, eval_subvolume_fullfnames, eval_labels_data, training_set, hash_function, evaluation, tuning_results_file, features_table, train_sample_seed, eval_sample_seed, parameter_selection_seed, checkpoint) = _loading_data(
                    preproc_volume_fullfname, train_subvolume_dir, train_label_dirs,
                    eval_subvolume_dir, eval_label_dirs, config, search_results_fullfname,
                    best_result_fullfname, parameter_selection_timeout, use_features_table,
                    features_table_fullfname, train_sample_seed, eval_sample_seed, parameter_selection_seed, checkpoint_fullfname, checkpoint_namespace,
                    reset_checkpoint, checkpoint_init, max_processes, max_batch_memory, use_gpu, listener
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
                (best_config,) = _tuning(config_data, segmenter, slice_shape, slice_size, full_volume, train_subvolume_slice_labels, volume_slice_indexes_in_train_subvolume, eval_subvolume_slice_labels, volume_slice_indexes_in_eval_subvolume, training_set, evaluation, parameter_selection_timeout, tuning_results_file, features_table, train_sample_seed, eval_sample_seed, checkpoint, max_processes, max_batch_memory, listener, extra_result_col_names, extra_result_col_values)
            listener.log_output('Tuned')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(6, 'Saving best config')
            listener.log_output(times.get_timestamp())
            listener.log_output('Saving best config')
            with times.Timer() as timer:
                () = _saving_best_config(best_result_fullfname, best_config, listener)
            listener.log_output('Saved')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()

        return best_config
    except Exception as ex:
        listener.error_output(str(ex))
        if debug_mode:
            raise
    finally:
        if full_volume is not None:
            full_volume.close()
