import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import volumes

#########################################
def visualise(in_array, sigma):
    min_value = np.min(in_array)
    max_value = np.max(in_array)

    downsample_kernel = downscales.GaussianDownsampleKernel(sigma=sigma)
    out_array1 = downscales.downscale(in_array, downsample_kernel, 1, remove_pad=False)
    out_array2 = downscales.downscale(in_array, downsample_kernel, 2, remove_pad=False)

    (fig, ax) = plt.subplots(1, 1)
    kernel = downsample_kernel.get_kernel(1)
    ax.set_title('Sum of kernel values: {}'.format(kernel.sum()))
    ax.matshow(kernel[kernel.shape[0]//2,:,:], cmap='gray')
    fig.show()

    (fig, axs) = plt.subplots(3, 3, figsize=(12,8))

    for (row, array) in enumerate([ in_array, out_array1, out_array2 ]):
        for col in range(3):
            img = np.rot90([
                    array[array.shape[0]//2,:,:],
                    array[:,array.shape[1]//2,:],
                    array[:,:,array.shape[2]//2],
                ][col])
            img = regions.get_neighbourhood_array_2d(img, [img.shape[0]//2, img.shape[1]//2], 50, {0,1,2})
            axs[row,col].matshow(img, cmap='gray', vmin=min_value, vmax=max_value)
            axs[row,col].set_xlabel([ 'row', 'slice', 'row' ][col])
            axs[row,col].set_ylabel([ 'column', 'column', 'slice' ][col])
            axs[row,col].set_xticks([], [])
            axs[row,col].set_yticks([], [])
    
    fig.tight_layout()
    fig.show()

#########################################

full_volume = volumes.FullVolume(os.path.join('..', 'example_volume', 'output', 'preprocess', 'volume.hdf'))
full_volume.load()
in_array = full_volume.get_scale_array(0)[:]

visualise(in_array, sigma=np.sqrt(2))

input('Press enter to exit.')
