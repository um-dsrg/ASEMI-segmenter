'''High level functions that act as commands for the segmentation tool.'''

import math
import os
import numpy as np
from asemi_segmenter.lib import datas
from asemi_segmenter.lib import files
from asemi_segmenter.lib import times
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import evaluations


#########################################
class ProgressListener(object):
    '''Class for listening to the progress of segmenter commands.'''

    #########################################
    def __init__(self):
        '''Empty constructor.'''
        pass

    #########################################
    def log_output(self, text):
        '''
        Listener for each line of text logging the command's activity.

        :param str text: The line of log text.
        '''
        pass

    #########################################
    def error_output(self, text):
        '''
        Listener for fatal errors.

        :param str text: The error message.
        '''
        pass

    #########################################
    def overall_progress_start(self, total):
        '''
        Listener for initialising a progress bar for the whole command's activity.

        :param int total: The total number of stages in the command.
        '''
        pass

    #########################################
    def overall_progress_update(self, curr, status):
        '''
        Listener for the start of a stage in the whole command's activity.

        :param int curr: The number of the stage that has started.
        :param str status: A short text description of the stage that has started.
        '''
        pass

    #########################################
    def overall_progress_end(self):
        '''
        Listener for the destruction of the progress bar for the whole command's activity.
        '''
        pass

    #########################################
    def current_progress_start(self, start, total):
        '''
        Listener for initialising a progress bar for a sub-stage.

        :param int start: The starting iteration in the progress.
            Normally 0, but can be more if a checkpoint is resumed.
        :param int total: The total number of iterations in the sub-stage.
        '''
        pass

    #########################################
    def current_progress_update(self, curr):
        '''
        Listener for the completion of an iteration in the sub-stage.

        :param int curr: The number of the iteration that has completed.
        '''
        pass

    #########################################
    def current_progress_end(self):
        '''
        Listener for the destruction of the progress bar for the sub-stage.
        '''
        pass


#########################################
def preprocess(
        volume_dir, config, result_data_fullfname,
        checkpoint_fullfname, restart_checkpoint,
        max_processes, max_batch_memory, listener=ProgressListener()
    ):
    '''
    Preprocess the slice images of a volume into a single HDF file usable by the other commands.

    :param str volume_dir: The path to the directory containing the slice images of the volume.
    :param config: The configuration to use when preprocessing (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the preprocess configuration.
    :type config: str or dict
    :param str result_data_fullfname: Full file name (with path) to HDF file to create.
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
            listener.overall_progress_start(5)

            listener.log_output('Starting preprocessing process.')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading inputs')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading inputs...')
            with times.Timer() as timer:
                listener.log_output('> Loading volume directory.')
                volume_data = datas.load_volume_dir(volume_dir)

                listener.log_output('> Loading config file.')
                if isinstance(config, str):
                    (config_data, num_downsamples, downsample_filter, hash_function) = \
                        datas.load_preprocess_config_file(config)
                else:
                    (config_data, num_downsamples, downsample_filter, hash_function) = \
                        datas.load_preprocess_config_data(config)

                listener.log_output('> Checking result data file name.')
                if result_data_fullfname is not None:
                    datas.check_preprocessed_filename(result_data_fullfname)

                listener.log_output('> Checking checkpoint file name.')
                if checkpoint_fullfname is not None:
                    datas.check_checkpoint_filename(checkpoint_fullfname)

                listener.log_output('> Initialising.')
                slice_shape = volume_data.shape
                volume_fullfnames = volume_data.fullfnames
                hash_function.init(slice_shape, seed=0)
                full_volume = datas.FullVolume(result_data_fullfname)
                checkpoint = datas.CheckpointManager(
                    'preprocess',
                    checkpoint_fullfname,
                    restart_checkpoint
                    )
            listener.log_output('Input loaded.')
            listener.log_output('Duration: {}'.format(
                times.get_readable_duration(timer.duration)
                ))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Creating empty data file')
            listener.log_output(times.get_timestamp())
            listener.log_output('Creating empty data file...')
            with times.Timer() as timer:
                with checkpoint.apply('empty_data_file') as skip:
                    if skip is not None:
                        listener.log_output('> Skipped as was found ready.')
                        raise skip
                    volume_shape = (len(volume_fullfnames), *slice_shape)
                    full_volume.create(config_data, volume_shape)
                full_volume.load()
            listener.log_output('Empty data file created.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(3, 'Dumping slices into data file')
            listener.log_output(times.get_timestamp())
            listener.log_output('Dumping slices into data file...')
            with times.Timer() as timer:
                with checkpoint.apply('dump_slices') as skip:
                    if skip is not None:
                        listener.log_output('> Skipped as was found ready.')
                        raise skip
                    listener.current_progress_start(0, len(volume_fullfnames))

                    def post_processor(result):
                        '''
                        Function defining what to do with the result of each processor.
                        '''
                        (volume_slice_index, img_data) = result
                        full_volume.get_scale_array(0)[volume_slice_index, :, :] = img_data

                    arrayprocs.parallel_processer(
                        lambda volume_slice_index, volume_fullfname: (
                            volume_slice_index,
                            datas.load_image(volume_fullfname)
                            ),
                        enumerate(volume_fullfnames),
                        post_processor=post_processor,
                        n_jobs=max_processes,
                        extra_params=(),
                        progress_listener=lambda num_ready, num_new: (
                            listener.current_progress_update(num_ready)
                            ))
                    listener.current_progress_end()
            listener.log_output('Slices dumped.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(4, 'Downscaling volume')
            listener.log_output(times.get_timestamp())
            listener.log_output('Downscaling volume...')
            with times.Timer() as timer:
                context_needed = downsample_filter.get_context_needed(1)
                for scale in range(1, num_downsamples+1):
                    listener.log_output('> Downscaling volume to scale {}.'.format(scale))
                    with checkpoint.apply('downscale_{}'.format(scale)) as skip:
                        if skip is not None:
                            listener.log_output('>> Skipped as was found ready.')
                            raise skip
                        best_block_shape = arrayprocs.get_optimal_block_size(
                            full_volume.get_scale_array(scale-1).shape,
                            full_volume.get_dtype(),
                            context_needed,
                            max_processes,
                            max_batch_memory,
                            implicit_depth=False
                            )
                        listener.current_progress_start(
                            0, arrayprocs.get_num_blocks_in_data(
                                full_volume.get_scale_array(scale-1).shape,
                                best_block_shape,
                                context_needed
                                )
                            )
                        downscales.downscale_in_blocks(
                            full_volume.get_scale_array(scale-1),
                            full_volume.get_scale_array(scale),
                            best_block_shape,
                            downsample_filter,
                            1,
                            n_jobs=max_processes,
                            progress_listener=lambda num_ready, num_new:\
                                listener.current_progress_update(num_ready)
                            )
                        listener.current_progress_end()
            listener.log_output('Volume downscaled.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(5, 'Hashing volume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing volume slices...')
            with times.Timer() as timer:
                with checkpoint.apply('hashing_slices') as skip:
                    if skip is not None:
                        listener.log_output('> Skipped as was found ready.')
                        raise skip
                    listener.current_progress_start(0, len(volume_fullfnames))
                    for volume_slice_index in range(len(volume_fullfnames)):
                        img_data = full_volume.get_scale_array(0)[volume_slice_index, :, :]
                        full_volume.get_hashes_array()[volume_slice_index, :] = \
                            hash_function.apply(img_data)
                        listener.current_progress_update(volume_slice_index+1)
                    listener.current_progress_end()
                del hash_function
            listener.log_output('Slices hashed.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done.')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()
    except datas.DataException as ex:
        listener.error_output(str(ex))
    finally:
        if full_volume is not None:
            full_volume.close()


#########################################
def train(
        preproc_volume_fullfname, subvolume_dir, label_dirs, config,
        result_model_fullfname, trainingset_file_fullfname,
        checkpoint_fullfname, restart_checkpoint,
        max_processes, max_batch_memory, listener=ProgressListener()
    ):
    '''
    Train a classifier model to segment volumes based on manually labelled slices.

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
    :param result_model_fullfname: Full file name (with path) to pickle file to create. If None
        then model will be returned instead of saved.
    :type result_model_fullfname: str or None
    :param str checkpoint_fullfname: Full file name (with path) to checkpoint pickle.
    :param checkpoint_fullfname: Full file name (with path) to checkpoint pickle. If None then no
        checkpointing is used.
    :type checkpoint_fullfname: str or None
    :param bool restart_checkpoint: Whether to ignore checkpoint and start process from beginning.
    :param int max_processes: The maximum number of processes to use concurrently.
    :param float max_batch_memory: The maximum number of gigabytes to use between all processes.
    :param ProgressListener listener: The command's progress listener.
    :return: If result_model_fullfname was None, returns the trained model as a dictionary.
        See user guide for description of the model dictionary.
    :rtype: None or dict
    '''
    full_volume = None
    training_set = None
    try:
        with times.Timer() as full_timer:
            listener.overall_progress_start(6)

            listener.log_output('Starting training process.')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data...')
            with times.Timer() as timer:
                listener.log_output('> Loading full volume data file.')
                datas.check_preprocessed_filename(preproc_volume_fullfname)
                full_volume = datas.FullVolume(preproc_volume_fullfname)
                full_volume.load()
                (_, _, _, hash_function) = full_volume.get_config()

                listener.log_output('> Loading subvolume directory.')
                subvolume_data = datas.load_volume_dir(subvolume_dir)

                listener.log_output('> Loading labels.')
                labels_data = []
                for (i, label_dir) in enumerate(label_dirs):
                    listener.log_output('>> Loading label {} directory.'.format(i+1))
                    label_data = datas.load_label_dir(label_dir)
                    labels_data.append(label_data)

                listener.log_output('> Loading config file.')
                if isinstance(config, str):
                    (config_data, featuriser, classifier) = \
                        datas.load_train_config_file(config, full_volume)
                else:
                    (config_data, featuriser, classifier) = \
                        datas.load_train_config_data(config, full_volume)
                classifier.n_jobs = max_processes

                listener.log_output('> Checking result model file name.')
                if result_model_fullfname is not None:
                    datas.check_model_filename(result_model_fullfname)

                listener.log_output('> Checking checkpoint file name.')
                if checkpoint_fullfname is not None:
                    datas.check_checkpoint_filename(checkpoint_fullfname)

                listener.log_output('> Initialising.')
                datas.validate_annotation_data(full_volume, subvolume_data, labels_data)
                feature_size = featuriser.get_feature_size()
                context_needed = featuriser.get_context_needed()
                slice_shape = full_volume.get_shape()[1:]
                slice_size = slice_shape[0]*slice_shape[1]
                subvolume_fullfnames = subvolume_data.fullfnames
                labels = sorted(label_data.name for label_data in labels_data)
                hash_function.init(slice_shape, seed=0)
                checkpoint = datas.CheckpointManager(
                    'train',
                    checkpoint_fullfname,
                    restart_checkpoint
                    )
                training_set = datas.TrainingSet(trainingset_file_fullfname)
                sample_size_per_label = config_data['training_set']['sample_size_per_label']

                del subvolume_data
                del label_data
            listener.log_output('Input loaded.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Reserving training set space')
            listener.log_output(times.get_timestamp())
            listener.log_output('Reserving training set space...')
            with times.Timer() as timer:
                if trainingset_file_fullfname is not None:
                    with checkpoint.apply('reserving_trainingset') as skip:
                        if skip is not None:
                            listener.log_output('> Continuing use of existing training set.')
                            raise skip
                        training_set.create(slice_size*len(subvolume_fullfnames), feature_size)
                    training_set.load()
                else:
                    training_set.create(slice_size*len(subvolume_fullfnames), feature_size)
            listener.log_output('Training set space reserved.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(3, 'Hashing subvolume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing subvolume slices...')
            with times.Timer() as timer:
                listener.current_progress_start(0, len(subvolume_fullfnames))
                subvolume_hashes = np.empty(
                    (len(subvolume_fullfnames), hash_function.hash_size),
                    full_volume.get_hashes_dtype())
                for (i, fullfname) in enumerate(subvolume_fullfnames):
                    img_data = datas.load_image(fullfname)
                    subvolume_hashes[i, :] = hash_function.apply(img_data)
                    listener.current_progress_update(i+1)
                listener.current_progress_end()
                volume_slice_indexes_in_subvolume = datas.get_volume_slice_indexes_in_subvolume(
                    full_volume.get_hashes_array()[:], subvolume_hashes  #Load the hashes eagerly.
                    )
                listener.log_output('> Subvolume to volume file name mapping found:')
                for (subvolume_index, volume_index) in enumerate(
                        volume_slice_indexes_in_subvolume
                    ):
                    listener.log_output('>> {} -> volume slice #{}'.format(
                        subvolume_fullfnames[subvolume_index], volume_index+1
                        ))
                del hash_function
            listener.log_output('Slices hashed.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(4, 'Constructing training set')
            listener.log_output(times.get_timestamp())
            listener.log_output('Constructing training set...')
            with times.Timer() as timer:
                listener.log_output('> Constructing labels.')
                with checkpoint.apply('contructing_labels') as skip:
                    if skip is not None:
                        listener.log_output('> Skipped as was found ready.')
                        raise skip
                    training_set.get_labels_array()[:] = datas.load_labels(labels_data)

                listener.log_output('> Constructing features.')
                best_block_shape = arrayprocs.get_optimal_block_size(
                    slice_shape,
                    full_volume.get_dtype(),
                    context_needed,
                    max_processes,
                    max_batch_memory,
                    implicit_depth=True
                    )
                with checkpoint.apply('constructing_features') as skip:
                    if skip is not None:
                        listener.log_output('> Skipped as was found ready.')
                        raise skip
                    start = checkpoint.get_next_to_process('constructing_features_prog')
                    listener.current_progress_start(start, len(subvolume_fullfnames))
                    for (i, volume_slice_index) in enumerate(volume_slice_indexes_in_subvolume):
                        if i < start:
                            continue
                        with checkpoint.apply('constructing_features_prog'):
                            featuriser.featurise(
                                full_volume.get_scale_arrays(featuriser.get_scales_needed()),
                                slice_index=volume_slice_index,
                                block_rows=best_block_shape[0],
                                block_cols=best_block_shape[0],
                                output=training_set.get_features_array(),
                                output_start_index=i*slice_size,
                                n_jobs=max_processes
                                )
                        listener.current_progress_update(i+1)
                    listener.current_progress_end()
                del subvolume_fullfnames
                del volume_slice_indexes_in_subvolume
                del featuriser
            listener.log_output('Training set constructed.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(5, 'Training segmenter')
            listener.log_output(times.get_timestamp())
            listener.log_output('Training segmenter...')
            with times.Timer() as timer:
                if sample_size_per_label == -1:
                    mask = datas.get_subvolume_slice_label_mask(training_set.get_labels_array())
                    classifier.fit(
                        training_set.get_features_array()[mask],
                        training_set.get_labels_array()[mask]
                        )
                else:
                    listener.log_output('> Sampling training set.')
                    new_training_set = training_set.get_sample(sample_size_per_label, seed=0)
                    listener.log_output('> Training.')
                    classifier.fit(
                        new_training_set.get_features_array(),
                        new_training_set.get_labels_array()
                        )
            listener.log_output('Segmenter trained.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(6, 'Saving model')
            listener.log_output(times.get_timestamp())
            listener.log_output('Saving model...')
            with times.Timer() as timer:
                model = {'model': classifier, 'labels': labels, 'config': config_data}
                datas.save_model(result_model_fullfname, model)
            listener.log_output('Model saved.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done.')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()

        if result_model_fullfname is None:
            return model
        return None
    except datas.DataException as e:
        listener.error_output(str(e))
    finally:
        if full_volume is not None:
            full_volume.close()
        if training_set is not None and trainingset_file_fullfname is not None:
            training_set.close()


#########################################
def evaluate(
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

            listener.log_output('Starting evaluation process.')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading inputs')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading inputs...')
            with times.Timer() as timer:
                listener.log_output('> Loading full volume data file.')
                datas.check_preprocessed_filename(preproc_volume_fullfname)
                full_volume = datas.FullVolume(preproc_volume_fullfname)
                full_volume.load()
                (_, _, _, hash_function) = full_volume.get_config()

                listener.log_output('> Loading model file.')
                if isinstance(model, str):
                    (config_data, labels, featuriser, classifier) = \
                        datas.load_model_file(model, full_volume)
                else:
                    (config_data, labels, featuriser, classifier) = \
                        datas.load_model_data(model, full_volume)
                classifier.n_jobs = max_processes
                del config_data

                listener.log_output('> Loading subvolume directory.')
                subvolume_data = datas.load_volume_dir(subvolume_dir)

                listener.log_output('> Loading labels.')
                labels_data = []
                for (i, label_dir) in enumerate(label_dirs):
                    listener.log_output('>> Loading label {} directory.'.format(i+1))
                    label_data = datas.load_label_dir(label_dir)
                    labels_data.append(label_data)

                listener.log_output('> Checking result file name.')
                if results_fullfname is not None:
                    datas.check_evaluation_results_filename(results_fullfname)

                listener.log_output('> Checking checkpoint file name.')
                if checkpoint_fullfname is not None:
                    datas.check_checkpoint_filename(checkpoint_fullfname)

                listener.log_output('> Initialising.')
                datas.validate_annotation_data(full_volume, subvolume_data, labels_data)
                context_needed = featuriser.get_context_needed()
                slice_shape = full_volume.get_shape()[1:]
                slice_size = slice_shape[0]*slice_shape[1]
                subvolume_fullfnames = subvolume_data.fullfnames
                evaluation_labels = sorted(label_data.name for label_data in labels_data)
                if evaluation_labels != labels:
                    raise ValueError('Labels in evaluation directory do not match labels in model.')
                evaluation_results_file = datas.EvaluationResultsFile(results_fullfname)
                hash_function.init(slice_shape, seed=0)
                checkpoint = datas.CheckpointManager(
                    'evaluate',
                    checkpoint_fullfname,
                    restart_checkpoint
                    )

                del subvolume_data
                del label_data
                del evaluation_labels
            listener.log_output('Input loaded.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Hashing subvolume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing subvolume slices...')
            with times.Timer() as timer:
                listener.current_progress_start(0, len(subvolume_fullfnames))
                subvolume_hashes = np.empty(
                    (len(subvolume_fullfnames), hash_function.hash_size),
                    full_volume.get_hashes_dtype()
                    )
                for (i, fullfname) in enumerate(subvolume_fullfnames):
                    img_data = datas.load_image(fullfname)
                    subvolume_hashes[i, :] = hash_function.apply(img_data)
                    listener.current_progress_update(i+1)
                listener.current_progress_end()
                volume_slice_indexes_in_subvolume = datas.get_volume_slice_indexes_in_subvolume(
                    full_volume.get_hashes_array()[:], subvolume_hashes  #Load the hashes eagerly.
                    )
                listener.log_output('> Subvolume to volume file name mapping found:')
                for (subvolume_index, volume_index) in enumerate(volume_slice_indexes_in_subvolume):
                    listener.log_output('>> {} -> volume slice #{}'.format(
                        subvolume_fullfnames[subvolume_index], volume_index+1
                        ))
            del hash_function
            listener.log_output('Slices hashed.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(3, 'Constructing labels dataset')
            listener.log_output(times.get_timestamp())
            listener.log_output('Constructing labels dataset...')
            with times.Timer() as timer:
                subvolume_slice_labels = datas.load_labels(labels_data)
            listener.log_output('Labels dataset constructed.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(4, 'Evaluating')
            listener.log_output(times.get_timestamp())
            listener.log_output('Evaluating...')
            listener.log_output('-')
            with times.Timer() as timer:
                results = dict()
                with checkpoint.apply('create_results_file') as skip:
                    if skip is not None:
                        listener.log_output('> Continuing use of existing results file.')
                        listener.log_output('-')
                        raise skip
                    evaluation_results_file.create(labels)
                best_block_shape = arrayprocs.get_optimal_block_size(
                    slice_shape,
                    full_volume.get_dtype(),
                    context_needed,
                    max_processes,
                    max_batch_memory,
                    implicit_depth=True
                    )
                for (i, (subvolume_fullfname, volume_slice_index)) in enumerate(
                        zip(subvolume_fullfnames, volume_slice_indexes_in_subvolume)
                    ):
                    listener.log_output('> Evaluating {} ({:.2%}).'.format(
                        subvolume_fullfname, (i+1)/len(subvolume_fullfnames)
                        ))
                    with checkpoint.apply('evaluating_{}'.format(volume_slice_index)) as skip:
                        if skip is not None:
                            listener.log_output('>> Skipped as was found ready.')
                            listener.log_output('-')
                            raise skip
                        with times.Timer() as sub_timer:
                            with times.Timer() as sub_timer_featuriser:
                                (slice_features, _) = featuriser.featurise(
                                    full_volume.get_scale_arrays(featuriser.get_scales_needed()),
                                    slice_index=volume_slice_index,
                                    block_rows=best_block_shape[0],
                                    block_cols=best_block_shape[0],
                                    n_jobs=max_processes
                                    )

                            with times.Timer() as sub_timer_classifier:
                                prediction = classifier.predict_proba(slice_features)
                                prediction = np.argmax(prediction, axis=1)
                            del slice_features
                            prediction = prediction.reshape(slice_shape)

                            slice_labels = subvolume_slice_labels[i*slice_size:(i+1)*slice_size]
                            slice_labels = slice_labels.reshape(slice_shape)

                            ious = evaluations.get_intersection_over_union(
                                prediction, slice_labels, len(labels)
                                )
                            del prediction
                            del slice_labels
                            for (label, iou) in zip(labels, ious):
                                if iou is not None:
                                    listener.log_output('>> {}: {:.3%}'.format(label, iou))
                            evaluation_results_file.append(
                                subvolume_fullfname, ious,
                                sub_timer_featuriser.duration, sub_timer_classifier.duration
                                )
                            results[subvolume_fullfname] = ious

                        listener.log_output('   Duration: {} (featurisation: {}, ' \
                            'classification: {})'.format(
                                times.get_readable_duration(sub_timer.duration),
                                times.get_readable_duration(sub_timer_featuriser.duration),
                                times.get_readable_duration(sub_timer_classifier.duration)
                                ))
                        listener.log_output('-')

            listener.log_output('Evaluated.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done.')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()

        if results_fullfname is None:
            return results
        return None
    except datas.DataException as e:
        listener.error_output(str(e))
    finally:
        if full_volume is not None:
            full_volume.close()


#########################################
def segment(
        model, preproc_volume_fullfname, soft_segmentation, results_dir,
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
    :param bool soft_segmentation: Whether to output the segmented slices as grayscale images
        consisting of graded levels of yes and no (high intensity pixels versus low intensity
        pixels or to output black and white images.
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

            listener.log_output('Starting segmentation process.')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading inputs')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading inputs...')
            with times.Timer() as timer:
                listener.log_output('> Loading full volume data file.')
                datas.check_preprocessed_filename(preproc_volume_fullfname)
                full_volume = datas.FullVolume(preproc_volume_fullfname)
                full_volume.load()
                full_volume.get_config()  # Just to validate.

                listener.log_output('> Loading model file.')
                if isinstance(model, str):
                    (config_data, labels, featuriser, classifier) = \
                        datas.load_model_file(model, full_volume)
                else:
                    (config_data, labels, featuriser, classifier) = \
                        datas.load_model_data(model, full_volume)
                del config_data
                classifier.n_jobs = max_processes

                listener.log_output('> Checking results directory.')
                datas.check_segmentation_results_directory(results_dir)

                listener.log_output('> Checking checkpoint file name.')
                if checkpoint_fullfname is not None:
                    datas.check_checkpoint_filename(checkpoint_fullfname)

                listener.log_output('> Initialising.')
                context_needed = featuriser.get_context_needed()
                slice_shape = full_volume.get_shape()[1:]
                checkpoint = datas.CheckpointManager(
                    'segment',
                    checkpoint_fullfname,
                    restart_checkpoint
                    )
            listener.log_output('Input loaded.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Segmenting')
            listener.log_output(times.get_timestamp())
            listener.log_output('Segmenting...')
            with times.Timer() as timer:
                for label in labels:
                    files.mkdir(os.path.join(results_dir, label))

                num_digits_in_filename = math.ceil(math.log10(full_volume.get_shape()[0]+1))
                best_block_shape = arrayprocs.get_optimal_block_size(
                    slice_shape,
                    full_volume.get_dtype(),
                    context_needed,
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
                            (slice_features, _) = featuriser.featurise(
                                full_volume.get_scale_arrays(featuriser.get_scales_needed()),
                                slice_index=volume_slice_index,
                                block_rows=best_block_shape[0],
                                block_cols=best_block_shape[0],
                                n_jobs=max_processes
                                )

                            prediction = classifier.predict_proba(slice_features)
                            if not soft_segmentation:
                                classes = np.argmax(prediction, axis=1)
                                prediction[:, :] = 0.0
                                prediction[np.arange(prediction.shape[0]), classes] = 1.0
                                del classes
                            prediction = np.round(prediction*255).astype(np.uint8)
                            del slice_features
                            prediction = prediction.reshape((*slice_shape, len(labels)))

                            for (label_index, label) in enumerate(labels):
                                datas.save_image(
                                    os.path.join(
                                        results_dir,
                                        label,
                                        '{}_{:0>{}}.png'.format(
                                            label,
                                            volume_slice_index+1,
                                            num_digits_in_filename
                                            )
                                        ),
                                    prediction[:, :, label_index]
                                    )
                            del prediction
                        listener.current_progress_update(volume_slice_index+1)
                    listener.current_progress_end()
            listener.log_output('Volume segmented.')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

        listener.log_output('Done.')
        listener.log_output('Entire process duration: {}'.format(
            times.get_readable_duration(full_timer.duration)
            ))
        listener.log_output(times.get_timestamp())

        listener.overall_progress_end()
    except datas.DataException as e:
        listener.error_output(str(e))
    finally:
        if full_volume is not None:
            full_volume.close()
