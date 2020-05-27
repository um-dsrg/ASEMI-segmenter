'''Preprocess command.'''

import json
from asemi_segmenter import listeners
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import checkpoints
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import hashfunctions
from asemi_segmenter.lib import images
from asemi_segmenter.lib import times
from asemi_segmenter.lib import validations
from asemi_segmenter.lib import volumes


#########################################
def _loading_data(
        volume_dir, config, result_data_fullfname,
        checkpoint_fullfname, checkpoint_namespace, reset_checkpoint,
        checkpoint_init, max_processes, max_batch_memory, listener
    ):
    '''Loading data stage.'''
    listener.log_output('> Volume')
    listener.log_output('>> {}'.format(volume_dir))
    volume_data = volumes.load_volume_dir(volume_dir)
    slice_shape = volume_data.shape
    volume_fullfnames = volume_data.fullfnames

    listener.log_output('> Config')
    if isinstance(config, str):
        listener.log_output('>> {}'.format(config))
        validations.check_filename(config, '.json', True)
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = config
    validations.validate_json_with_schema_file(config_data, 'preprocess.json')
    downsample_filter = downscales.load_downsamplekernel_from_config(config_data['downsample_filter'])
    hash_function = hashfunctions.load_hashfunction_from_config(config_data['hash_function'])

    listener.log_output('> Result')
    listener.log_output('>> {}'.format(result_data_fullfname))
    validations.check_filename(result_data_fullfname, '.hdf', False)
    full_volume = volumes.FullVolume(result_data_fullfname)

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

    return (config_data, full_volume, volume_fullfnames, slice_shape, downsample_filter, hash_function, checkpoint)


#########################################
def _creating_empty_data_file(
        config_data, full_volume, volume_fullfnames, slice_shape, checkpoint, listener
    ):
    '''Creating empty data file stage.'''
    with checkpoint.apply('empty_data_file') as skip:
        if skip is not None:
            listener.log_output('> Skipped as was found checkpointed')
            raise skip
        volume_shape = (len(volume_fullfnames), *slice_shape)
        full_volume.create(config_data, volume_shape)
    full_volume.load()

    return ()


#########################################
def _dumping_slices_into_data_file(
        full_volume, volume_fullfnames, max_processes, checkpoint, listener
    ):
    '''Dumping slices into data file stage.'''
    with checkpoint.apply('dump_slices') as skip:
        if skip is not None:
            listener.log_output('> Skipped as was found checkpointed')
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
                images.load_image(volume_fullfname)
                ),
            enumerate(volume_fullfnames),
            post_processor=post_processor,
            n_jobs=max_processes,
            extra_params=(),
            progress_listener=lambda num_ready, num_new: (
                listener.current_progress_update(num_ready)
                ))
        listener.current_progress_end()

    return ()


#########################################
def _downscaling_volume(
        config_data, full_volume, downsample_filter, max_processes, max_batch_memory, checkpoint, listener
    ):
    '''Downscaling volume stage.'''
    context_needed = downsample_filter.get_context_needed(1)
    for scale in range(1, config_data['num_downsamples']+1):
        listener.log_output('> Downscaling volume to scale {}'.format(scale))
        with checkpoint.apply('downscale_{}'.format(scale)) as skip:
            if skip is not None:
                listener.log_output('>> Skipped as was found checkpointed')
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

    return ()


#########################################
def _hashing_volume_slices(
    full_volume, volume_fullfnames, hash_function, checkpoint, listener
    ):
    '''Hashing volume slices stage.'''
    with checkpoint.apply('hashing_slices') as skip:
        if skip is not None:
            listener.log_output('> Skipped as was found checkpointed')
            raise skip
        listener.current_progress_start(0, len(volume_fullfnames))
        for volume_slice_index in range(len(volume_fullfnames)):
            img_data = full_volume.get_scale_array(0)[volume_slice_index, :, :]
            full_volume.get_hashes_array()[volume_slice_index, :] = \
                hash_function.apply(img_data)
            listener.current_progress_update(volume_slice_index+1)
        listener.current_progress_end()

    return ()


#########################################
def main(
        volume_dir,
        config,
        result_data_fullfname,
        checkpoint_fullfname=None,
        checkpoint_namespace='preprocess',
        reset_checkpoint=False,
        checkpoint_init=dict(),
        max_processes=-1,
        max_batch_memory=1,
        listener=listeners.ProgressListener(),
        debug_mode=False
    ):
    '''
    Preprocess the slice images of a volume into a single HDF file usable by the other commands.

    :param str volume_dir: The path to the directory containing the slice images of the volume.
    :param config: The configuration to use when preprocessing (can be either a path to a
        json file containing the configuration or a dictionary specifying the configuration
        directly). See user guide for description of the preprocess configuration.
    :type config: str or dict
    :param str result_data_fullfname: Full file name (with path) to HDF file to create.
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
            listener.overall_progress_start(5)

            listener.log_output('Starting preprocessing process')
            listener.log_output('')

            ###################

            listener.overall_progress_update(1, 'Loading data')
            listener.log_output(times.get_timestamp())
            listener.log_output('Loading data')
            with times.Timer() as timer:
                (config_data, full_volume, volume_fullfnames, slice_shape, downsample_filter, hash_function, checkpoint) = _loading_data(
                    volume_dir, config, result_data_fullfname,
                    checkpoint_fullfname, checkpoint_namespace, reset_checkpoint,
                    checkpoint_init, max_processes, max_batch_memory, listener
                    )
            listener.log_output('Data loaded')
            listener.log_output('Duration: {}'.format(
                times.get_readable_duration(timer.duration)
                ))
            listener.log_output('')

            ###################

            listener.overall_progress_update(2, 'Creating empty data file')
            listener.log_output(times.get_timestamp())
            listener.log_output('Creating empty data file')
            with times.Timer() as timer:
                () = _creating_empty_data_file(config_data, full_volume, volume_fullfnames, slice_shape, checkpoint, listener)
            listener.log_output('Empty data file created')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(3, 'Dumping slices into data file')
            listener.log_output(times.get_timestamp())
            listener.log_output('Dumping slices into data file')
            with times.Timer() as timer:
                () = _dumping_slices_into_data_file(full_volume, volume_fullfnames, max_processes, checkpoint, listener)
            listener.log_output('Slices dumped')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(4, 'Downscaling volume')
            listener.log_output(times.get_timestamp())
            listener.log_output('Downscaling volume')
            with times.Timer() as timer:
                () = _downscaling_volume(config_data, full_volume, downsample_filter, max_processes, max_batch_memory, checkpoint, listener)
            listener.log_output('Volume downscaled')
            listener.log_output('Duration: {}'.format(times.get_readable_duration(timer.duration)))
            listener.log_output('')

            ###################

            listener.overall_progress_update(5, 'Hashing volume slices')
            listener.log_output(times.get_timestamp())
            listener.log_output('Hashing volume slices')
            with times.Timer() as timer:
                () = _hashing_volume_slices(full_volume, volume_fullfnames, hash_function, checkpoint, listener)
            listener.log_output('Slices hashed')
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