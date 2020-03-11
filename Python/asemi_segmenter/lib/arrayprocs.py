import os
import math
import numpy as np
import joblib
import itertools
import sys
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import times

#########################################
def get_num_processes(num_processes):
    cpu_count = joblib.cpu_count()
    if num_processes > 0:
        return min(cpu_count, num_processes)
    elif num_processes < 0:
        return min(cpu_count, cpu_count + 1 + num_processes)
    else:
        raise ValueError('num_processes cannot be 0.')

#########################################
def parallel_processer(processor, processor_params, total_params, post_processor=lambda result:None, n_jobs=1, extra_params=(), progress_listener=lambda ready, total, duration:None):
    n_jobs = get_num_processes(n_jobs)
    params_visited = 0
    with joblib.Parallel(n_jobs, max_nbytes=None) as parallel: #If max_nbytes is not None then you get out of space errors.
        while True:
            with times.Timer() as timer:
                parallel_batch = itertools.islice(processor_params, n_jobs)
                batch_size = 0
                for result in parallel(joblib.delayed(processor)(*params, *extra_params) for params in parallel_batch):
                    post_processor(result)
                    batch_size += 1
                if batch_size == 0:
                    return
                params_visited += batch_size
            progress_listener(params_visited, total_params, timer.duration)

#########################################
def get_num_blocks_in_data(data_shape, block_shape, context_needed):
    return np.prod([ math.ceil(d/(b - 2*context_needed)) for (d,b) in zip(data_shape, block_shape) ])

#########################################
def get_optimal_block_size(data_shape, data_dtype, context_needed, num_processes, max_batch_memory_gb, implicit_depth=False):
    '''
    Given that arrays are faster to divide across the first dimension than the last (because of locality of reference), the division across processes is made across the first dimension. It is important that if that dimension is not divisible by the number of processes then the largest remainder should be chosen in order to keep the blocks as equally sized as possible. The other dimensions are scaled only to make the block fit within the given maximum memory.
    '''
    space_available = math.floor(max_batch_memory_gb*(1024**3)/np.dtype(data_dtype).itemsize) #Number of elements.
    
    def fairest_divisor(numerator, parts_needed):
        '''
        The smallest divisor such that ceil(numerator/divisor) == parts_needed will either not leave any remainder or leave the largest remainder.
        
        When numerator = 10, different divisors give the following parts_needed:
        
        >>> for d in range(1,10+1):
        >>>    print(d, math.ceil(10/d), 10%d)
        
        divisor parts remainder
        1       10    0
        2       5     0
        3       4     1
        4       3     2
        5       2     0
        6       2     4
        7       2     3
        8       2     2
        9       2     1
        10      1     0
        
        We need to find the smallest divisor that gives the desired number of parts as that will have a remainder of 0 or maximum.
        
        Notice that not all parts are obtainable, in which case we will use the first divisor that gives less than or equal to parts as desired.
        
        Implementation is very naive. Will be replaced with a more mathematically informative implementation later.
        '''
        for divisor in range(1, numerator+1):
            parts_found = math.ceil(numerator/divisor)
            if parts_found <= parts_needed:
                return divisor
    
    #Initialise to a single block the size of the data.
    parts_in_each_side = [ 1 for _ in range(len(data_shape)) ]
    contextless_block_shape = list(data_shape)
    
    #Divide the first dimension of the data into as many parts as there are processes, if available.
    parts_in_each_side[0] = min(num_processes, data_shape[0])
    contextless_block_shape = [ fairest_divisor(s, p) for (s,p) in zip(data_shape, parts_in_each_side) ]
    
    def space_needed(contextless_block_shape):
        block_size = np.prod([ (2*context_needed + side) for side in contextless_block_shape ])
        if implicit_depth:
            block_size *= 2*context_needed + 1
        return block_size*num_processes
    
    def num_blocks(contextless_block_shape):
        return np.prod([ math.ceil(d/b) for (d,b) in zip(data_shape, contextless_block_shape) ])
    
    #Continue reducing block size until a feasible size is reached. Reduction is obtained by putting more parts in a side of the data shape.
    while (
            space_needed(contextless_block_shape) > space_available or
            num_blocks(contextless_block_shape) < num_processes
        ):
        argmax_side = np.argmax(contextless_block_shape) #Greedily pick the largest side of the block to reduce.
        if contextless_block_shape[argmax_side] > 1:
            parts_in_each_side[argmax_side] += 1
        else:
            raise ValueError('Maximum batch memory is too small or there are too many processes to even process a piece of the volume.')
        
        contextless_block_shape = [ fairest_divisor(s, p) for (s,p) in zip(data_shape, parts_in_each_side) ]
        
    block_shape = tuple((2*context_needed + side) for side in contextless_block_shape)
    return block_shape

#########################################
def process_array_in_blocks(in_array_scales, out_array, processor, block_shape, scales=None, in_ranges=None, context_size=0, pad_value=0, n_jobs=1, extra_params=(), progress_listener=lambda ready, total, duration:None):
    '''
    Transform an array into a new array such that the input array is processed in blocks that can be processed concurrently. 'in_array_scales' is a dictionary of 3D arrays at different scales such that in_array_scales[0] gives the original array, in_array_scales[1] gives the array at half size, etc. (2^-scale). If no scales are available then just put the array in a dictionary like this: { 0: array }. If only specific scales are needed then you can use 'scales' to specify a list of scales to consider, otherwise all scales in 'in_array_scales' will be used. Only a single output array is returned and the 'block_shape' and 'context_size' is the same among all blocks and scales.
    
    - 'processor' is a function that takes in a block from 'in_array_scales' and returns a new block together with the slice tuple in which to write the block in 'out_array'. Information about each requested scale is presented to the processor. Expected signature of processor is:
        params: ({
                scale: {
                        incontext_slices_wrt_whole,
                        incontext_slices_wrt_range,
                        contextless_slices_wrt_whole,
                        contextless_slices_wrt_range,
                        contextless_slices_wrt_block,
                        contextless_shape,
                        block,
                    },
            }, *extra_params)
        return: (transformed_block, out_slice_tuple)
    - 'block_shape' is the shape of the numpy array to pass to the processor (or smaller if close to the edge of the array, although the context will always be the same size and only the payload size will change), including context (in order to make it easy to control memory use). Each shape side must be larger than 2*'context_size' in order to be valid (otherwise no payload would fit).
    - 'in_ranges' is a list of slices in every dimension of 'in_array_scales' to actually process.
    - 'context_size' is used when the processor requires using a context window to work. By setting a context_size, each block will be padded with 'context_size' adjacent values (if far enough from the edge) plus padding with 'pad_value' if necessary. Note that you can use the 'processors''s 'contextless_slices_wrt_block' to stip away the context.
    - 'extra_params' is a tuple of extra parameters to pass to processor that are constant (in order to avoid creating external dependencies during multiprocessing).
    
    Example:
        context_size 0, block_shape [5]
        in_array_scales[0] 123456789 => blocks [12345,6789]
                                       context:
        
        context_size 1, block_shape [5]
        in_array_scales[0]  123456789 => blocks [01234,34567,67890]
                                        context: ^   ^ ^   ^ ^   ^
        
        context_size 1, block_shape [6]
        in_array_scales[0]  123456789 => blocks [012345,456789,890]
                                        context: ^    ^ ^    ^ ^ ^
        
        context_size 2, block_shape [5]
        in_array_scales[0]  123456789 => blocks [00123,01234,12345,23456,34567,45678,56789,67890,78900]
                                        context: ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^ ^^
    '''
    if scales is None:
        scales = sorted(in_array_scales.keys())
    if any((scale not in in_array_scales) for scale in scales):
        raise ValueError('Requested scales that are not available in in_array_scales.')
    if 0 not in in_array_scales:
        raise ValueError('Input array must include scale 0.')
    if any(len(in_array_scales[scale].shape) != 3 for scale in in_array_scales):
        raise ValueError('Input arrays must be 3 dimensional.')
    if any(in_array_scales[scale].shape != downscales.predict_new_shape(in_array_scales[0].shape, scale) for scale in in_array_scales):
        raise ValueError('Input array shapes must be according to their scale.'.format())
    if any(l <= 2*context_size for l in block_shape):
        raise ValueError('One or more sides of the block shape is too small to contain even one voxel in context (must all be at least {}).'.format(2*context_size+1))
    if in_ranges is None:
        in_ranges_scaled = {
                scale: [
                        downscales.downscale_slice(slice(0, l), scale)
                        for l in in_array_scales[0].shape
                    ]
                for scale in { 0 } | set(scales)
            }
    else:
        if len(in_ranges) != len(in_array_scales[0].shape):
            raise ValueError('Number of ranges must be equal to number of dimensions in input array.')
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
                for scale in { 0 } | set(scales)
            }
    
    def get_processor_params():
        steps = [ l - 2*context_size for l in block_shape ]
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
        
        for contextless_slice_start in range(in_ranges_scaled[0][0].start, in_ranges_scaled[0][0].stop, steps[0]):
            for contextless_row_start in range(in_ranges_scaled[0][1].start, in_ranges_scaled[0][1].stop, steps[1]):
                for contextless_col_start in range(in_ranges_scaled[0][2].start, in_ranges_scaled[0][2].stop, steps[2]):
                    params = { scale: dict() for scale in scales }
                    for scale in scales:
                        contextless_starts = [
                                downscales.downscale_pos(contextless_slice_start, scale),
                                downscales.downscale_pos(contextless_row_start, scale),
                                downscales.downscale_pos(contextless_col_start, scale),
                            ]
                        contextless_shape = [
                                min(max_l, boundery.stop - i)
                                for (max_l, boundery, i) in zip(steps, in_ranges_scaled[scale], contextless_starts)
                            ]
                        incontext_slices_wrt_whole = [
                                slice(i - context_size, i+l+context_size)
                                for (i, l) in zip(contextless_starts, contextless_shape)
                            ]
                        params[scale]['incontext_slices_wrt_whole'] = tuple(incontext_slices_wrt_whole)
                        params[scale]['incontext_slices_wrt_range'] = tuple(
                                slice(w.start - r.start, w.stop - r.start)
                                for (w, r) in zip(incontext_slices_wrt_whole, in_ranges_scaled[scale])
                            )
                        params[scale]['contextless_slices_wrt_whole'] = tuple(
                                slice(i, i+l)
                                for (i, l) in zip(contextless_starts, contextless_shape)
                            )
                        params[scale]['contextless_slices_wrt_range'] = tuple(
                                slice(w.start - r.start, w.stop - r.start)
                                for (w, r) in zip(params[scale]['contextless_slices_wrt_whole'], in_ranges_scaled[scale])
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
                        
                    yield [ params ]
    
    total_num_blocks = get_num_blocks_in_data([ (s.stop - s.start) for s in in_ranges_scaled[0] ], block_shape, context_size)
    
    def post_processor(result):
        if result is not None:
            (transformed_block, out_slices) = result
            out_array[out_slices] = transformed_block
    
    parallel_processer(
            processor,
            get_processor_params(),
            total_num_blocks,
            post_processor=post_processor,
            n_jobs=n_jobs,
            extra_params=extra_params,
            progress_listener=progress_listener
        )
    return out_array

#########################################
def process_array_in_blocks_single_slice(in_array_scales, out_array, processor, block_shape, slice_index, scales=None, in_ranges=None, context_size=0, pad_value=0, n_jobs=1, extra_params=(), progress_listener=lambda ready, total, duration:None):
    '''
    Version of process_array_in_blocks that is meant to work on a single slice in a volume (with context from adjacent slices).
    Note that the first dimension contextless_slices_wrt_whole and contextless_slices_wrt_block are an integer rather than a slice so as to access the 2D array of interest.
    '''
    if scales is None:
        scales = sorted(in_array_scales.keys())
    if any((scale not in in_array_scales) for scale in scales):
        raise ValueError('Requested scales that are not available in in_array_scales.')
    if 0 not in in_array_scales:
        raise ValueError('Input array must include scale 0.')
    if any(len(in_array_scales[scale].shape) != 3 for scale in in_array_scales):
        raise ValueError('Input arrays must be 3 dimensional.')
    if any(in_array_scales[scale].shape != downscales.predict_new_shape(in_array_scales[0].shape, scale) for scale in in_array_scales):
        raise ValueError('Input array shapes must be according to their scale.')
    if any(l <= 2*context_size for l in block_shape):
        raise ValueError('One or more sides of the block shape is too small to contain even one voxel in context (must all be at least {}).'.format(2*context_size+1))
    
    new_in_ranges = [ slice(slice_index, slice_index+1) ] + (list(in_ranges) if in_ranges is not None else [ slice(None), slice(None) ])
    
    def new_processor(params, contextless_slice_index, context_size, *extra_params):
        return processor(
                {
                        scale: {
                                'incontext_slices_wrt_whole': params[scale]['incontext_slices_wrt_whole'],
                                'incontext_slices_wrt_range': params[scale]['incontext_slices_wrt_range'],
                                'contextless_slices_wrt_whole': (downscales.downscale_pos(contextless_slice_index, scale),)+params[scale]['contextless_slices_wrt_whole'][1:],
                                'contextless_slices_wrt_range': (downscales.downscale_pos(contextless_slice_index, scale) - params[scale]['contextless_slices_wrt_range'][0].start,)+params[scale]['contextless_slices_wrt_range'][1:],
                                'contextless_slices_wrt_block': (context_size,)+params[scale]['contextless_slices_wrt_block'][1:],
                                'contextless_shape': params[scale]['contextless_shape'][1:],
                                'block': params[scale]['block'],
                            }
                        for scale in params
                    },
                *extra_params
            )
    
    new_block_shape = tuple([ 2*context_size + 1 ] + list(block_shape))
    
    return process_array_in_blocks(in_array_scales, out_array, new_processor, new_block_shape, scales, new_in_ranges, context_size, pad_value, n_jobs, (slice_index, context_size)+extra_params, progress_listener)
