'''
Tools for processing large arrays.

-----------------------
Explanation of blocks:

A block is a piece of a large array that fits in memory, can be processed
in isolation, and its transformation placed in a resulting array. A block usually includes some
extra data from the array as context for the inner data in the block. For example, if a very large
image is to be blurred with a Gaussian blur using a convolution filter, and the image does not fit
in memory in its entirety, then the image must be loaded into memory one block at a time, each
block be blurred in isolation and the blurred block be placed in the corresponding location in
the resulting blurred image. To do this, the block must include some extra pixels at the edges as
context in order to be able to correctly blur the edges of the payload. Here is an example using a
1D image:

Image: abcdefghij

Block size: 5 with a context of 1 (payload is the 3 elements in the middle of the block, context is
    1 element from each side of the block)

Padding value: 0 (this is used for when there needs to be a context but the block is at the edge of
    the image and so there are no pixels to use for context)

Therefore blocks from the image are:
    block 1:  0abcd
    pay load:  ^^^
    context:  ^   ^

    block 2:  cdefg
    pay load:  ^^^
    context:  ^   ^

    block 3:  fghij
    pay load:  ^^^
    context:  ^   ^

    block 4:  ij0
    pay load:  ^
    context:  ^ ^

Note how putting all the pay loads together in sequence will form the original image. Each of these
blocks is processed separately from the rest using a fixed pool of concurrent processes. In the
program, this method is applied to 3D images which would be too large to fit in memory.
'''

import math
import itertools
import numpy as np
import joblib
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import downscales

#########################################
def get_num_processes(num_processes):
    '''
    Convert the requested number of processes to a canonical amount.

    The amount of processes specified can be negative, in which case the amount of actual processes
    would be num_CPUs - num_processes + 1, that is using -1 as num_processes uses all available
    CPUs. This function also trims the amount if it is greater than the amount of CPUs available.

    :param int num_processes: The requested number of processes to use.
    :return: The actual number of processes that will be used.
    :rtype: int
    '''
    cpu_count = joblib.cpu_count()
    if num_processes > 0:
        return min(cpu_count, num_processes)
    elif num_processes < 0:
        return min(cpu_count, cpu_count + 1 + num_processes)
    else:
        raise ValueError('num_processes cannot be 0.')

#########################################
def parallel_processer(
        processor, processor_params, post_processor=lambda result: None, n_jobs=1,
        extra_params=(), progress_listener=lambda num_ready, num_new: None
    ):
    '''
    Process a list of parameter sets with a single processing function in parallel.

    The way this function works is that a fixed number of processes is used to run the same number
    of parameter sets. These processes have to all wait for each other to finish, at which point
    a fresh batch of parameter sets will be processed. This is to make it possible to use a single
    shared progress bar which, rather than having the processes individually update it, will only
    be updated by main thread. After every batch of processes is done processing their respective
    parameter sets, the main thread will call the progress listener with the amount of parameter
    sets that have been processed up to now in total.

    :param callable processor: A function that takes in a parameter set (from `processor_params`)
        and returns a result.
    :param iter processor_params: A generator or list of tuples, where the tuples are to be passed
        to `processor`.
    :param callable post_processor: A function that takes in each result from `processor`
        individually and does something with the result. Result of this function is ignored. This
        is useful when `post_processor` has some side effect.
    :param int n_jobs: Number of processes to use concurrently. See `get_num_processes` for an
        explanation.
    :param tuple extra_params: A tuple of extra parameters to pass to `processor` with every
        parameter set. This is meant to be always the same and is concatenated to the end of each
        tuple in `processor_params`.
    :param callable progress_listener: A function that receives the number of parameter sets that
        have been processed in total and the number of new parameter sets just processed.
    '''
    n_jobs = get_num_processes(n_jobs)
    params_visited = 0

    #If max_nbytes is not None then you get out of space errors.
    with joblib.Parallel(n_jobs, max_nbytes=None) as parallel:
        while True:
            parallel_batch = itertools.islice(processor_params, n_jobs)
            batch_size = 0
            for res in parallel(
                    joblib.delayed(processor)(*params, *extra_params)
                    for params in parallel_batch
                ):
                post_processor(res)
                batch_size += 1
            if batch_size == 0:
                return
            params_visited += batch_size
            progress_listener(params_visited, batch_size)

#########################################
def get_num_blocks_in_data(data_shape, block_shape, context_needed):
    '''
    Get the number of blocks in a given shape of data.

    See module-level docstring for an explanation of blocks.

    :param tuple data_shape: The shape of the data to break into blocks.
    :param tuple block_shape: The shape of the block to use (with context included).
    :param int context_needed: The thickness of the context around the edges of the block.
    :return: The number of blocks.
    :rtype: int
    '''
    return np.prod([
        math.ceil(d/(b - 2*context_needed))
        for (d, b) in zip(data_shape, block_shape)
        ])

#########################################
def get_optimal_block_size(
        data_shape, data_dtype, context_needed, num_processes, max_batch_memory_gb,
        num_implicit_slices=None, feature_size=None, feature_dtype=None
    ):
    '''
    Get the block shape that fits in a given memory size and results in fast processing.

    The block shape is not meant to be user defined directly. Instead, the user should only
    specify the amount of memory that can be used by the block and then a block shape that fits
    is found. Further constraints are that the block must divide the data into a number of parts
    that is divisible or almost divisible by the number of processes and is also fast to break up
    into said parts. Given that arrays are faster to divide across the first dimension than the
    last (because of locality of reference), the data is first divided at the first dimension. The
    first step is to break the data at the first dimension into the number of processes available.
    It is important that if that dimension is not divisible by the number of processes then the
    largest remainder should be chosen in order to keep the blocks as equally sized as possible.
    The other dimensions are divided only to make each block fit within the given maximum memory.

    Blocks should also be multiples of 32 in order to optimise performance on GPUs.

    :param tuple data_shape: The shape of the data to break down.
    :param numpy_datatype data_dtype: The numpy data type of the data to break down.
    :param int context_needed: The thickness of the context on the edges of the block.
    :param int num_processes: The number of processes to run in parallel.
    :param float max_batch_memory_gb: The maximum number of gigabytes (1024^3) to allow a block to
        use.
    :param int num_implicit_slices: If the given data is 2D but needs to have its context be in 3D,
        the number of slices to process in each block (without the context) is entered here.
        This is for when a volume is processed slice by slice and so the payload of interest is a
        single slice (or number of slices) but the adjacent slices should be used for context.
        If None then no context is added if data is 2D.
    :param int feature_size: The number of features that each voxel in the block will transform to.
        If the block is not being used for featurisation, leave as None.
    :param numpy_datatype feature_dtype: The data type of the feature vector that will come out.
        If the block is not being used for featurisation, leave as None.
    :return: The block shape (with context included).
    :rtype: tuple
    '''
    #Get the number of data elements that can fit in memory.
    in_space_available = math.floor(max_batch_memory_gb*(1024**3)/np.dtype(data_dtype).itemsize)
    if feature_dtype is not None:
        out_space_available = math.floor(max_batch_memory_gb*(1024**3)/np.dtype(feature_dtype).itemsize)
    else:
        out_space_available = None

    def get_next_block_size(data_size, start_block_size, blocks_needed):
        '''
        Break the data into a number of blocks by finding a good block size to break with.
        We need to find a block size such that ceil(data_size/block_size) >= blocks_needed
        and is a multiple of 32.

        Different block sizes can give the same required number of blocks. The ideal one
        will either not leave any remainder or will leave the largest remainder. The
        largest remainder is good if none of the remainders are 0 because that will make
        the blocks as balanced as possible.

        Example:

        When data_size = 11, different block sizes give the following number of blocks:

        >>> for block_size in range(1, 11+1):
        >>>    print(block_size, math.ceil(11/block_size), 11%block_size)

        block_size blocks remainder
        11          1     0
        10          2     1
         9          2     2
         8          2     3
         7          2     4
         6          2     5
         5          3     1
         4          3     3
         3          4     2
         2          6     1
         1         11     0

        Notice how there are several block sizes that give a number of blocks equal to 2, but the
        block size 6 gives the largest remainder which should be chosen. The smaller the block
        size, the better the remainder (eventually becoming zero, if available).

        Notice also that not all block numbers are obtainable, in which case we will use the first
        block size that gives the next larger number of blocks. This is because we need the number
        of blocks to increase with every step of the algorithm, so we cannot return less blocks
        than requested.

        start_block_size is the largest block size to start searching from and will be decreased
        in order for the number of blocks to grow.
        '''
        #Make start_block_size a multiple of 32.
        if start_block_size%32 > 0:
            start_block_size -= start_block_size%32

        for block_size in range(start_block_size, 32-1, -32):
            blocks_found = math.ceil(data_size/block_size)
            if blocks_found > blocks_needed:
                block_size += 32 #Go back one step.
                if math.ceil(data_size/block_size) == blocks_needed:
                    return block_size
                else:
                    #If we're here then the number of blocks obtained is worse than what
                    #we need and we should return to the step we got back from.
                    return block_size - 32

        return 32

    def in_space_needed(contextless_block_shape):
        '''The amount of space needed for a given block shape in input memory.'''
        block_size = np.prod([(2*context_needed + side) for side in contextless_block_shape]).tolist()
        if num_implicit_slices is not None:
            block_size *= 2*context_needed + num_implicit_slices
        return block_size*num_processes

    def out_space_needed(contextless_block_shape):
        '''The amount of space needed for a given block shape in output memory.'''
        block_size = np.prod(contextless_block_shape).tolist()*feature_size
        if num_implicit_slices is not None:
            block_size *= num_implicit_slices
        return block_size*num_processes

    def side_num_blocks(axis, contextless_block_shape):
        '''The number of blocks in a side of the data resulting from a given block shape.'''
        return math.ceil(data_shape[axis]/contextless_block_shape[axis])

    def total_num_blocks(contextless_block_shape):
        '''The total number of blocks in the data resulting from a given block shape.'''
        return np.prod([math.ceil(d/b) for (d, b) in zip(data_shape, contextless_block_shape)]).tolist()

    #Decrease the sides of the block in multiples of 32 until a feasible size is reached.
    contextless_block_shape = list(data_shape)
    while (
            in_space_needed(contextless_block_shape) > in_space_available or
            (
                out_space_needed(contextless_block_shape) > out_space_available
                if out_space_available is not None else False
                ) or
            total_num_blocks(contextless_block_shape) < num_processes
        ):
        #Pick a side of the block to reduce.
        #Pick greedily such that the data shape is broken into a balanced number of
        #blocks per side and is reducable (must be at least 32 elements).
        #Preference is given to the first axis of the block.
        axis = 0
        for i in range(1, len(data_shape)):
            if (
                    side_num_blocks(i, contextless_block_shape) < side_num_blocks(axis, contextless_block_shape)
                    and contextless_block_shape[i] > 32
                ):
                axis = i

        #If no decreasable side is found then ignore number of side blocks.
        if contextless_block_shape[axis] <= 32:
            axis = np.argmax(contextless_block_shape)

        #If no decreasable side is found then stop optimisation.
        if contextless_block_shape[axis] <= 32:
            break

        contextless_block_shape[axis] = get_next_block_size(
            data_shape[axis],
            contextless_block_shape[axis],
            side_num_blocks(axis, contextless_block_shape) + 1
            )

    if in_space_needed(contextless_block_shape) > in_space_available:
        raise ValueError('Not enough memory to even store the smallest possible block size.')
    if out_space_available is not None and out_space_needed(contextless_block_shape) > out_space_available:
        raise ValueError('Not enough memory to even store the smallest possible block size when featurised.')

    block_shape = tuple((2*context_needed + side) for side in contextless_block_shape)
    return block_shape

#########################################
def process_array_in_blocks(
        in_array_scales, out_array, processor, block_shape, scales=None, in_ranges=None,
        context_size=0, pad_value=0, n_jobs=1, extra_params=(),
        progress_listener=lambda num_ready, num_new: None
    ):
    '''
    Transform a 3D array into another array using blocks.

    The input array can be presented in different scales but only a single output array is
    returned and the block shape and context size are the same among all scales.

    :param dict in_array_scales: A dictionary of 3D arrays at different scales such that
        in_array_scales[0] gives the original array, in_array_scales[1] gives the array at
        half size, etc. (2^-scale). If no scales are available then just put the array in a
        dictionary like this: { 0: array }.
    :param nump.ndarray out_array: The array that is to contain the output (can be any
        dimensionality). The reason for requiring a preconstructed output array is to be memory
        efficient and avoid concatenating a bunch of separate arrays together.
    :param callable processor: A function that takes in a block from 'in_array_scales' and returns
        a new block together with the tuple index in which to write the block in 'out_array'. The
        tuple index consists of Python slices specifying the destination area e.g.
        (slice(0, 3), slice(0, 3)) (note that two slices means that the output array is 2D).
        Information about the input array at each scale is presented to the processor. Expected
        signature of processor is:
        processor(
            params: (
                {
                    scale: {
                        'incontext_slices_wrt_whole': _,
                        'incontext_slices_wrt_range': _,
                        'contextless_slices_wrt_whole': _,
                        'contextless_slices_wrt_range': _,
                        'contextless_slices_wrt_block': _,
                        'contextless_shape': _,
                        'block': _,
                    },
                },
                *extra_params
            )
        ) -> result
        Where result is either None (to signal that the block should not affect the output array
        or be the tuple (transformed_block, out_slice_tuple). Note that extra_params is explained
        below.
    :param tuple block_shape: The shape of the numpy array to pass to the processor (or smaller if
        close to the edge of the array, although the context will always be the same size and only
        the payload size will change), including context (in order to make it easier to control
        memory use). Each shape side must be larger than 2*context_size in order to be valid
        (otherwise no payload would fit).
    :param set scales: If only specific scales are needed then you can use this parameter to
        specify a list of scales to consider, otherwise all scales in in_array_scales will be used.
    :param list in_ranges: A list of Python slices in every dimension of in_array_scales to
        actually process.
    :param int context_size: This is used when the processor requires using a context window to
        work. By setting a context_size, each block will be padded with this amount of adjacent
        values. Note that you can use the processors's contextless_slices_wrt_block to stip away
        the context.
    :param int pad_value: If using a context size and the current block is too close to the edge,
        then this value is used in place of values outside of the array (over the edge).
    :param int n_jobs: Number of processes to run concurrently.
    :param tuple extra_params: The extra parameters to pass to the processor.
    :param callable progress_listener: A function that receives information about the progress of
        the whole array processing.
    :return: A reference to out_array.
    :rtype: numpy.ndarray
    '''
    if scales is None:
        scales = sorted(in_array_scales.keys())
    if any((scale not in in_array_scales) for scale in scales):
        raise ValueError('Requested scales that are not available in in_array_scales.')
    if 0 not in in_array_scales:
        raise ValueError('Input array must include scale 0.')
    if any(len(in_array_scales[scale].shape) != 3 for scale in in_array_scales):
        raise ValueError('Input arrays must be 3 dimensional.')
    if any(
            in_array_scales[scale].shape != downscales.predict_new_shape(
                in_array_scales[0].shape, scale
                )
            for scale in in_array_scales
        ):
        raise ValueError('Input array shapes must be according to their scale.'.format())
    if any(l <= 2*context_size for l in block_shape):
        raise ValueError(
            'One or more sides of the block shape is too small to contain even one voxel in '
            'context (must all be at least {}).'.format(2*context_size+1)
            )
    if in_ranges is None:
        in_ranges_scaled = {
            scale: [
                downscales.downscale_slice(slice(0, l), scale)
                for l in in_array_scales[0].shape
                ]
            for scale in {0} | set(scales)
            }
    else:
        if len(in_ranges) != len(in_array_scales[0].shape):
            raise ValueError(
                'Number of ranges must be equal to number of dimensions in input array.'
                )
        in_ranges_scaled = {
            scale: [
                downscales.downscale_slice(
                    slice(
                        s.start if s.start is not None else 0,
                        s.stop if s.stop is not None else l
                        ),
                    scale
                    )
                for (l, s) in zip(in_array_scales[0].shape, in_ranges)
                ]
            for scale in {0} | set(scales)
            }

    def get_processor_params():
        '''A generator of parameter sets for the processor.'''
        steps = [l - 2*context_size for l in block_shape]
        params = {
            scale: {
                'incontext_slices_wrt_whole': None,
                'incontext_slices_wrt_range': None,
                'contextless_slices_wrt_whole': None,
                'contextless_slices_wrt_range': None,
                'contextless_slices_wrt_block': None,
                'contextless_shape': None,
                'block': None,
                }
            for scale in scales
            }

        for contextless_slice_start in range(
                in_ranges_scaled[0][0].start,
                in_ranges_scaled[0][0].stop,
                steps[0]
            ):
            for contextless_row_start in range(
                    in_ranges_scaled[0][1].start,
                    in_ranges_scaled[0][1].stop,
                    steps[1]
                ):
                for contextless_col_start in range(
                        in_ranges_scaled[0][2].start,
                        in_ranges_scaled[0][2].stop,
                        steps[2]
                    ):
                    params = {scale: dict() for scale in scales}
                    for scale in scales:
                        contextless_starts = [
                            downscales.downscale_pos(contextless_slice_start, scale),
                            downscales.downscale_pos(contextless_row_start, scale),
                            downscales.downscale_pos(contextless_col_start, scale),
                            ]
                        contextless_shape = [
                            min(max_l, boundery.stop - i)
                            for (max_l, boundery, i) in zip(
                                steps,
                                in_ranges_scaled[scale],
                                contextless_starts
                                )
                            ]
                        incontext_slices_wrt_whole = [
                            slice(i - context_size, i+l+context_size)
                            for (i, l) in zip(contextless_starts, contextless_shape)
                            ]
                        params[scale]['incontext_slices_wrt_whole'] = tuple(
                            incontext_slices_wrt_whole
                            )
                        params[scale]['incontext_slices_wrt_range'] = tuple(
                            slice(w.start - r.start, w.stop - r.start)
                            for (w, r) in zip(
                                incontext_slices_wrt_whole,
                                in_ranges_scaled[scale]
                                )
                            )
                        params[scale]['contextless_slices_wrt_whole'] = tuple(
                            slice(i, i+l)
                            for (i, l) in zip(contextless_starts, contextless_shape)
                            )
                        params[scale]['contextless_slices_wrt_range'] = tuple(
                            slice(w.start - r.start, w.stop - r.start)
                            for (w, r) in zip(
                                params[scale]['contextless_slices_wrt_whole'],
                                in_ranges_scaled[scale]
                                )
                            )
                        params[scale]['contextless_slices_wrt_block'] = tuple(
                            slice(context_size, context_size+l)
                            for l in contextless_shape
                            )
                        params[scale]['contextless_shape'] = tuple(contextless_shape)

                        scaled = in_array_scales[scale]
                        params[scale]['block'] = regions.get_subarray_3d(
                            scaled, *incontext_slices_wrt_whole, pad_value
                            )

                    yield [params]

    def post_processor(result):
        '''A post processor for the block transformations which puts them in the output array.'''
        if result is not None:
            (transformed_block, out_slices) = result
            out_array[out_slices] = transformed_block

    parallel_processer(
        processor,
        get_processor_params(),
        post_processor=post_processor,
        n_jobs=n_jobs,
        extra_params=extra_params,
        progress_listener=progress_listener
        )
    return out_array


#########################################
def process_array_in_blocks_slice_range(
        in_array_scales, out_array, processor, block_shape, slice_range, scales=None,
        in_ranges=None, context_size=0, pad_value=0, n_jobs=1, extra_params=(),
        progress_listener=lambda num_ready, num_new: None
    ):
    '''
    Version of process_array_in_blocks that is meant to work on a range of slices in a volume.

    This function works exactly like process_array_in_blocks but is tuned for the special case
    when a range of slices in a volume are being processed (with context from adjacent slices)
    at once (the block must include all the slices in the range).

    :param dict in_array_scales: As explained in process_array_in_blocks.
    :param nump.ndarray out_array: As explained in process_array_in_blocks.
    :param callable processor: As explained in process_array_in_blocks, but with
        contextless_slices_wrt_whole and contextless_slices_wrt_block being integers instead of
        Python slices.
    :param tuple block_shape: As explained in process_array_in_blocks but only for the
        rows and columns.
    :param slice slice_range: The python slice giving the range of the slices to process in the
        volume.
    :param set scales: As explained in process_array_in_blocks.
    :param list in_ranges: As explained in process_array_in_blocks.
    :param int context_size: As explained in process_array_in_blocks.
    :param int pad_value: As explained in process_array_in_blocks.
    :param int n_jobs: As explained in process_array_in_blocks.
    :param tuple extra_params: As explained in process_array_in_blocks.
    :param callable progress_listener: As explained in process_array_in_blocks.
    :return: A reference to out_array.
    :rtype: numpy.ndarray
    '''
    if scales is None:
        scales = sorted(in_array_scales.keys())
    if any((scale not in in_array_scales) for scale in scales):
        raise ValueError('Requested scales that are not available in in_array_scales.')
    if 0 not in in_array_scales:
        raise ValueError('Input array must include scale 0.')
    if any(len(in_array_scales[scale].shape) != 3 for scale in in_array_scales):
        raise ValueError('Input arrays must be 3 dimensional.')
    if any(
            in_array_scales[scale].shape != downscales.predict_new_shape(
                in_array_scales[0].shape, scale
                )
            for scale in in_array_scales
        ):
        raise ValueError('Input array shapes must be according to their scale.')
    if len(block_shape) != 2:
        raise ValueError('block_shape must be 2 values only.')
    if any(l <= 2*context_size for l in block_shape):
        raise ValueError(
            'One or more sides of the block shape is too small to contain even one voxel in '
            'context (must all be at least {}).'.format(2*context_size+1)
            )
    if slice_range.start is None or slice_range.stop is None:
        raise ValueError('slice_range must have defined start and stop values.')

    slice_range = slice(
        max(slice_range.start, 0),
        min(slice_range.stop, in_array_scales[0].shape[0])
        )

    new_in_ranges = [
        slice_range
        ] + (
            list(in_ranges) if in_ranges is not None else [slice(None), slice(None)]
        )

    new_block_shape = tuple(
        [2*context_size + (slice_range.stop - slice_range.start)] + list(block_shape)
        )

    return process_array_in_blocks(
        in_array_scales,
        out_array,
        processor,
        new_block_shape,
        scales,
        new_in_ranges,
        context_size,
        pad_value,
        n_jobs,
        extra_params,
        progress_listener
        )
