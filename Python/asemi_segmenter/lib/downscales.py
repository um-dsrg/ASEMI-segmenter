'''
Module for resizing arrays to a smaller size.

In order to resize an image to be half its size, every second pixel in each row and column needs to
be dropped. But this results in aliasing (https://en.wikipedia.org/wiki/Aliasing) which is solved
by first blurring the image in order to smoothen out high frequency changes between pixels.

An array is resized in non-positive powers of two, such as 2^0, 2^-1, 2^-2, etc. In order to
specify to which scale the array should be resized, a non-negative integer is used such that
given a scale of s, the array is resized to be 2^-s its original size. So resizing using a
scale of 0 leaves the array in its original size whilst a scale of 1 reduces the array to half its
size.
'''

import math
import os
import sys
import numpy as np
import scipy
import scipy.ndimage
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import validations


#########################################
def load_downsamplekernel_from_config(config):
    '''
    Load a downsample kernel from a configuration dictionary.

    :param dict config: Configuration of the downsample kernel.
    :return: A downsample kernel object.
    :rtype: DownsampleKernel
    '''
    validations.validate_json_with_schema_file(config, 'downsample_filter.json')

    if config['type'] == 'none':
        return NullDownsampleKernel()
    elif config['type'] == 'gaussian':
        sigma = config['params']['sigma']
        return GaussianDownsampleKernel(sigma)



#########################################
class DownsampleKernel(object):
    '''
    Base class for all downsample kernels.

    A downsample kernel is used to eliminate high frequencies from an image by blurring it. This
    allows the image to be reduced in size by half by just dropping every second element.
    '''

    #########################################
    def __init__(self, name):
        '''
        Constructor.

        :param str name: The name of the kernel.
        '''
        self.name = name

    #########################################
    def get_context_needed(self, scale):
        '''
        Get the amount of context needed around every element for the kernel to work.

        :param int scale: The scale of the data on which this kernel will be used on.
        :return: The amount of context needed.
        :rtype: int
        '''
        raise NotImplementedError()

    #########################################
    def get_kernel(self, scale, ndims=3):
        '''
        Get the actual kernel to convolve.

        :param int scale: The scale of the data on which this kernel will be used on.
        :param int ndims: The number of dimensions in the data on which this kernel will be used
            on.
        :return: The kernel as a numpy array.
        :rtype: numpy.ndarray
        '''
        raise NotImplementedError()


#########################################
class NullDownsampleKernel(DownsampleKernel):
    '''A downsample kernel that does nothing and exists only as a baseline.'''

    #########################################
    def __init__(self, name='null'):
        '''
        Constructor.

        :param str name: The name of the kernel.
        '''
        super().__init__(name)

    #########################################
    def get_context_needed(self, scale):
        '''
        Get the amount of context needed around every element for the kernel to work.

        :param int scale: The scale of the data on which this kernel will be used on (not used in
            this class).
        :return: Just returns zero.
        :rtype: int
        '''
        return 0

    #########################################
    def get_kernel(self, scale, ndims=3):
        '''
        Get the actual kernel to convolve.

        :param int scale: The scale of the data on which this kernel will be used on (not used in
            this class).
        :param int ndims: The number of dimensions in the data on which this kernel will be used
            on.
        :return: Just a single 1 in an ndims dimensional array.
        :rtype: numpy.ndarray
        '''
        return np.ones([1]*ndims, np.float32)


#########################################
class GaussianDownsampleKernel(DownsampleKernel):
    '''Gaussian blur: https://en.wikipedia.org/wiki/Gaussian_filter'''

    #########################################
    def __init__(self, sigma=math.sqrt(2), name='gaussian'):
        '''
        Constructor.

        :param float sigma: The sigma value of the gaussian function.
        :param str name: The name of the kernel.
        '''
        super().__init__(name)
        self.sigma = sigma
        self.num_sigmas_from_mean = 3

    #########################################
    def get_sigma(self, scale):
        '''
        Get the sigma value needed to resize an array to a given scale.

        :param int scale: The target scale.
        :return: The sigma value.
        :rtype: float
        '''
        #Given blurring needed for scaling by half is 'sigma', blurring needed for scaling half by half is sqrt(sigma^2 + sigma^2) = sigma*sqrt(2). A further half requires sigma*sqrt(2)*sqrt(2) and so on.
        if scale > 0:
            return self.sigma*(math.sqrt(2)**(scale-1))
        else:
            return 0

    #########################################
    def get_kernel_size(self, scale):
        '''
        Get the size of a side of the kernel filter needed in order to cover 3 standard deviations.

        :param int scale: The target scale for which the kernel shall be used.
        :return: The kernel size.
        :rtype: int
        '''
        if scale > 0:
            return int(math.ceil((self.num_sigmas_from_mean*self.get_sigma(scale))*2 + 1))
        else:
            return 1

    #########################################
    def get_context_needed(self, scale):
        '''
        Get the amount of context needed around every element for the kernel to work.

        :param int scale: The scale of the data on which this kernel will be used on.
        :return: The amount of context needed.
        :rtype: int
        '''
        if scale > 0:
            return int(math.ceil(self.get_kernel_size(scale)/2))
        else:
            return 0

    #########################################
    def get_kernel(self, scale, ndims=3):
        '''
        Get the actual kernel to convolve.

        :param int scale: The scale of the data on which this kernel will be used on.
        :param int ndims: The number of dimensions in the data on which this kernel will be used
            on.
        :return: The kernel as a numpy array.
        :rtype: numpy.ndarray
        '''
        if scale > 0:
            sigma = self.get_sigma(scale)
            xs = np.meshgrid(*[ np.linspace(-self.num_sigmas_from_mean*sigma, self.num_sigmas_from_mean*sigma, self.get_kernel_size(scale)) ]*ndims)

            two_sigma_sqr = 2*(sigma**2)
            kernel = np.prod([ np.exp(-(x**2)/two_sigma_sqr)/math.sqrt(math.pi*two_sigma_sqr) for x in xs ], axis=0)
            kernel = kernel/np.sum(kernel)

            return kernel
        else:
            return np.ones([1]*ndims, np.float32)


#########################################
def downscale_pos(position, scale):
    '''
    Find where a position (coordinate) moves to in a downsampled array.

    Given a position in a full sized array, find the corresponding position when the array is
    downscaled.

    :param position: The coordinate in the full sized array.
    :type position: list or tuple or int
    :param int scale: The scale of the target array.
    :return: The rescaled position.
    :rtype: list or tuple or int
    '''
    scale_factor = 2**scale
    if type(position) is list:
        return [ p//scale_factor for p in position ]
    elif type(position) is tuple:
        return tuple(p//scale_factor for p in position)
    else:
        return position//scale_factor


#########################################
def downscale_slice(slice_, scale):
    '''
    Given a range (Python slice) in the full sized array, get the corresponding resized range.

    :param slice slice_: The Python slice to resize.
    :param int scale: The scale to resize the range to.
    :return: The resized range.
    :rtype: slice
    '''
    scale_factor = 2**scale
    return slice(
            slice_.start//scale_factor if slice_.start is not None else None,
            (slice_.stop-1)//scale_factor+1 if slice_.stop is not None else None
        )


#########################################
def predict_new_shape(shape, scale):
    '''
    Get the new shape after an array is rescaled.

    :param tuple shape: The shape of the full sized array.
    :param int scale: The target scale of the array.
    :return: The new shape.
    :rtype: tuple
    '''
    if scale > 0:
        scale_factor = 2**scale
        return tuple(l//scale_factor + (l%scale_factor > 0) for l in shape)
    else:
        return shape


#########################################
def downscale(in_array, downsample_kernel, scale, remove_pad=False, trim_front=None):
    '''
    Resize an array to a given scale.

    When this is used for resizing a large array using blocks, the trim_front parameter becomes
    useful. If we're only keeping every fourth element in an array (to downscale it to a quarter
    of its original size) and we're using blocks of size 5, the following situation might happen
    (using a 1D array as an example):

    Array:   abcdefghijklmn
    To keep: a   e   i   m
    Blocks:  11111222223333

    Note how if we're dropping elements within the blocks rather than from the full array, then we
    cannot just keep every fourth element starting from the beginning of the block as the blocks
    and decimations are not in sync. By using trim_front we can trim the blocks to start from the
    first element to survive the decimation.

    :param numpy.ndarray in_array: The array to resize.
    :param DownsampleKernel downsample_kernel: The filter to downsample with.
    :param int scale: The scale to resize to.
    :param bool remove_pad: Whether to remove the array's edges that were used as context by the
        downsample kernel.
    :param list trim_front: The amount to trim from the front of each dimension in the array. See
        explanation above for more information.
    :return: The resized array.
    :rtype: numpy.ndarray
    '''
    if scale > 0:
        if trim_front is None:
            trim_front = [ 0 ]*len(in_array.shape)
        filtered = scipy.ndimage.convolve(in_array, downsample_kernel.get_kernel(scale, len(in_array.shape)), mode='constant', cval=0)
        context_needed = downsample_kernel.get_context_needed(scale)
        if remove_pad and context_needed > 0:
            downsampled = filtered[tuple(slice(context_needed + t, -context_needed, 2**scale) for (s, t) in zip(in_array.shape, trim_front))]
        else:
            downsampled = filtered[tuple(slice(t, None, 2**scale) for (s, t) in zip(in_array.shape, trim_front))]
        return downsampled
    else:
        return in_array


#########################################
def downscale_in_blocks(in_array, out_array, block_shape, downsample_kernel, scale, n_jobs=1, progress_listener=lambda num_ready, num_new:None):
    '''
    Like downscale but for use with very large arrays that need to be processed in blocks.

    :param numpy.ndarray in_array: The array to downscale (can be HDF file).
    :param numpy.ndarray out_array: The resulting resized array (can be HDF file). Use
        predict_new_shape to know what shape to use.
    :param tuple block_shape: The block shape to use.
    :param DownsampleKernel downsample_kernel: The downsample kernel to use.
    :param int scale: The scale to resize to.
    :param int n_jobs: The number of concurrent processes to use.
    :param callable progress_listener: A function that receives information about the progress of
        the downscale.
    :return: A reference to out_array.
    :rtype: numpy.ndarray
    '''
    def processor(params, downsample_kernel, scale):
        '''Processor to use for process_array_in_blocks.'''
        scale_factor = 2**scale
        context_needed = downsample_kernel.get_context_needed(scale)
        trim_front = [ math.ceil(s.start/scale_factor)*scale_factor - s.start for s in params[0]['contextless_slices_wrt_whole'] ]
        if any(l <= 2*context_needed + t for (l, t) in zip(params[0]['block'].shape, trim_front)):
            return None

        downsampled = downscale(params[0]['block'], downsample_kernel, scale, remove_pad=True, trim_front=trim_front)

        new_pos = []
        for (s, t, l) in zip(params[0]['contextless_slices_wrt_whole'], trim_front, downsampled.shape):
            start = downscale_pos(s.start + t, scale)
            stop = start + l
            new_pos.append(slice(start, stop))

        return (downsampled, tuple(new_pos))

    context_needed = downsample_kernel.get_context_needed(scale)
    if len(out_array.shape) != len(in_array.shape) or out_array.dtype != in_array.dtype:
        raise ValueError('Output array must be an array of the same type and number of dimensions as input array.')
    if out_array.shape != predict_new_shape(in_array.shape, scale):
        raise ValueError('Output array shape is not equal to the predicted shape of in_array after downscaling (array shape={}, predicted shape={}).'.format(out_array.shape, predict_new_shape(in_array.shape, scale)))

    return arrayprocs.process_array_in_blocks(
            { 0: in_array },
            out_array,
            processor,
            block_shape,
            context_size=context_needed,
            pad_value=0,
            n_jobs=n_jobs,
            extra_params=(downsample_kernel, scale),
            progress_listener=progress_listener
        )


#########################################
def grow_array(array, scale, axises=None, orig_shape=None):
    '''
    Grow an array to be back to the size it was before downscaling by repeating its values.

    This is useful for knowing which values in the original array is each value in the downscaled
    array standing in for. For example:

    Original array:   abcdefghijklmn
    Downscaled array: acegikm
    Grown array:      aacceeggiikkmm

    Note how when the grown array is compared to the original array we can see that 'a' is standing
    in for 'a' and 'b', 'c' is standing in for 'c' and 'd', etc.

    :param numpy.ndarray array: The downscaled array.
    :param int scale: The scale at which it was downscaled.
    :param list axises: The list of dimensions to grow (to leave some dimensions as-is).
    :param tuple orig_shape: The original array's shape in order to be able to trim any excess
        growth.
    :return: The grown array.
    :rtype: numpy.ndarray
    '''
    scale_factor = 2**scale

    if axises is None:
        axises = range(len(array.shape))
    for axis in axises:
        if orig_shape is None:
            array = np.repeat(array, scale_factor, axis=axis)
        else:
            pre_slices = [ slice(None) for _ in array.shape ]
            pre_slices[axis] = slice(0, int(np.ceil(orig_shape[axis]/scale_factor)))
            pre_slices = tuple(pre_slices)

            post_slices = [ slice(None) for _ in array.shape ]
            post_slices[axis] = slice(0, orig_shape[axis])
            post_slices = tuple(post_slices)

            array = np.repeat(
                    array[pre_slices],
                    scale_factor,
                    axis=axis
                )[post_slices]
    return array