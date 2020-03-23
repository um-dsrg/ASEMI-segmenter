import numpy as np
import scipy
import scipy.ndimage
import math
import os
import sys
from asemi_segmenter.lib import arrayprocs

#########################################
class DownsampleKernel(object):
    
    #########################################
    def __init__(self, name):
        self.name = name
        
    #########################################
    def get_context_needed(self, scale):
        raise NotImplementedError()
    
    #########################################
    def get_kernel(self, scale, ndims=3):
        raise NotImplementedError()

#########################################
class NullDownsampleKernel(DownsampleKernel):
    
    #########################################
    def __init__(self, name='null'):
        super().__init__(name)
    
    #########################################
    def get_context_needed(self, scale):
        return 0
    
    #########################################
    def get_kernel(self, scale, ndims=3):
        return np.ones([1]*ndims, np.float32)

#########################################
class GaussianDownsampleKernel(DownsampleKernel):
    
    #########################################
    def __init__(self, sigma=math.sqrt(2), name='gaussian'):
        super().__init__(name)
        self.sigma = sigma
        self.num_sigmas_from_mean = 3
    
    #########################################
    def get_sigma(self, scale):
        #Given blurring needed for scaling by half is 'sigma', blurring needed for scaling half by half is sqrt(sigma^2 + sigma^2) = sigma*sqrt(2). A further half requires sigma*sqrt(2)*sqrt(2) and so on.
        if scale > 0:
            return self.sigma*(math.sqrt(2)**(scale-1))
        else:
            return 0
    
    #########################################
    def get_kernel_size(self, scale):
        if scale > 0:
            return int(math.ceil((self.num_sigmas_from_mean*self.get_sigma(scale))*2 + 1))
        else:
            return 1
    
    #########################################
    def get_context_needed(self, scale):
        if scale > 0:
            return int(math.ceil(self.get_kernel_size(scale)/2))
        else:
            return 0
    
    #########################################
    def get_kernel(self, scale, ndims=3):
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
    '''Find corresponding position in a downscaled volume.'''
    scale_factor = 2**scale
    if type(position) is list:
        return [ p//scale_factor for p in position ]
    elif type(position) is tuple:
        return tuple(p//scale_factor for p in position)
    else:
        return position//scale_factor

#########################################
def downscale_slice(slice_, scale):
    '''Find equivalent range in a particular scale.'''
    scale_factor = 2**scale
    return slice(
            slice_.start//scale_factor if slice_.start is not None else None,
            (slice_.stop-1)//scale_factor+1 if slice_.stop is not None else None
        )

#########################################
def predict_new_shape(shape, scale):
    '''Find new shape in a particular scale.'''
    if scale > 0:
        scale_factor = 2**scale
        return tuple(l//scale_factor + (l%scale_factor > 0) for l in shape)
    else:
        return shape

#########################################
def downscale(in_array, downsample_kernel, scale, remove_pad=False, trim_front=None):
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
    
    def processor(params, downsample_kernel, scale):
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