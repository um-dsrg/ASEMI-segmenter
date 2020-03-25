import unittest
import numpy as np
import os
import sys
from asemi_segmenter.lib import arrayprocs
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import regions

#########################################
class ArrayProcs(unittest.TestCase):
    
    #########################################
    def test_process_array_in_blocks(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            scaled_data = { 0: (np.arange(7*7*7) + 1).reshape([7,7,7]) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            scaled_data[2] = downscales.downscale(scaled_data[0], downsample_kernel, 2)
            
            out = np.zeros_like(scaled_data[0])
            
            with self.assertRaises(ValueError):
                out[:,:,:] = 0
                arrayprocs.process_array_in_blocks(
                    scaled_data, out,
                    lambda params:(-params[0]['block'][params[0]['contextless_slices_wrt_block']], params[0]['contextless_slices_wrt_whole']),
                    block_shape=[2]*3, context_size=1, n_jobs=1
                )
            with self.assertRaises(ValueError):
                out[:,:,:] = 0
                arrayprocs.process_array_in_blocks(
                    scaled_data, out,
                    lambda params:(-params[0]['block'][params[0]['contextless_slices_wrt_block']], params[0]['contextless_slices_wrt_whole']),
                    block_shape=[2]*3, context_size=2, n_jobs=1
                )
            with self.assertRaises(ValueError):
                out[:,:,:] = 0
                arrayprocs.process_array_in_blocks(
                    scaled_data, out,
                    lambda params:(-params[0]['block'][params[0]['contextless_slices_wrt_block']], params[0]['contextless_slices_wrt_whole']),
                    block_shape=[4]*3, context_size=2, n_jobs=1
                )
            
            def test_block_processor(params, scaled_data, block_shape, context_size, scale):
                assert all(params[scale]['block'].shape[i] <= block_shape[i] for i in range(3)), '{}, {}, position={}, context_size={}, scale={}'.format(params[scale]['block'].shape, block_shape, params[scale]['incontext_slices_wrt_whole'], context_size, scale)
                np.testing.assert_equal(
                        regions.get_subarray_3d(scaled_data[scale], *params[scale]['incontext_slices_wrt_whole']),
                        params[scale]['block'],
                        'downsample_kernel={}, block_shape={}, context_size={}, scale={}'.format(downsample_kernel.name, block_shape, context_size, scale)
                    )
                np.testing.assert_equal(
                        params[scale]['block'][params[scale]['contextless_slices_wrt_block']],
                        scaled_data[scale][params[scale]['contextless_slices_wrt_whole']],
                        'downsample_kernel={}, block_shape={}, context_size={}, scale={}'.format(downsample_kernel.name, block_shape, context_size, scale)
                    )
                np.testing.assert_equal(
                        params[scale]['block'][params[scale]['contextless_slices_wrt_block']].shape,
                        params[scale]['contextless_shape'],
                        'downsample_kernel={}, block_shape={}, context_size={}, scale={}'.format(downsample_kernel.name, block_shape, context_size, scale)
                    )
                return (
                        -params[scale]['block'][params[scale]['contextless_slices_wrt_block']],
                        params[scale]['contextless_slices_wrt_whole']
                    )
            
            for scale in [ 0, 1, 2 ]:
                out = np.zeros_like(scaled_data[scale])
                
                for (block_shape, context_size, n_jobs) in [
                        ([1]*3, 0, 1),
                        ([2]*3, 0, 1),
                        ([3]*3, 1, 1),
                        ([3]*3, 1, 2),
                        ([6]*3, 2, 1),
                        ([20]*3, 2, 1),
                    ]:
                    context_size += downsample_kernel.get_context_needed(scale)
                    if block_shape[0] < 2*context_size + 1:
                        continue
                    
                    out[:,:,:] = 0
                    np.testing.assert_equal(
                            arrayprocs.process_array_in_blocks(
                                    scaled_data, out,
                                    test_block_processor,
                                    block_shape=block_shape, scales=[ scale ], context_size=context_size, n_jobs=n_jobs, extra_params=(scaled_data, block_shape, context_size, scale),
                                ),
                            -scaled_data[scale],
                            'downsample_kernel={}, scale={}, block_shape={}, context_size={}, n_jobs={}'.format(downsample_kernel.name, scale, block_shape, context_size, n_jobs)
                        )
        
        out = np.zeros([7,7,7,7*7*7], np.int32)
        
        out[:,:,:,:] = 0
        np.testing.assert_equal(
                arrayprocs.process_array_in_blocks(
                        { 0: scaled_data[0] }, out,
                        lambda params:(np.histogram(params[0]['block'], 7*7*7, (1, 7*7*7+1))[0], params[0]['contextless_slices_wrt_whole']+(slice(None),)),
                        block_shape=[3]*3, context_size=1, n_jobs=1
                    ),
                np.array([
                        [
                            [
                                np.histogram(regions.get_neighbourhood_array_3d(scaled_data[0], (slice_index, row, col), 1, {0,1,2}), 7*7*7, (1, 7*7*7+1))[0]
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ] for slice_index in range(scaled_data[0].shape[0])
                    ], np.int32)
            )
        
        out[:,:,:,:] = 0
        np.testing.assert_equal(
                arrayprocs.process_array_in_blocks(
                        { 0: scaled_data[0] }, out,
                        lambda params:(np.histogram(params[0]['block'], 7*7*7, (1, 7*7*7+1))[0], params[0]['contextless_slices_wrt_whole']+(slice(None),)),
                        block_shape=[5]*3, context_size=2, n_jobs=1
                    ),
                np.array([
                        [
                            [
                                np.histogram(regions.get_neighbourhood_array_3d(scaled_data[0], (slice_index, row, col), 2, {0,1,2}), 7*7*7, (1, 7*7*7+1))[0]
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ] for slice_index in range(scaled_data[0].shape[0])
                    ], np.int32)
            )
        
        out[:,:,:,:] = 0
        np.testing.assert_equal(
                arrayprocs.process_array_in_blocks(
                        { 0: scaled_data[0] }, out,
                        lambda params:
                                (
                                    np.array([
                                            [
                                                [
                                                    np.histogram(regions.get_neighbourhood_array_3d(params[0]['block'], (slice_index, row, col), 1, {0,1,2}), 7*7*7, (1, 7*7*7+1))[0]
                                                    for col in range(params[0]['contextless_slices_wrt_block'][2].start, params[0]['contextless_slices_wrt_block'][2].stop)
                                                ] for row in range(params[0]['contextless_slices_wrt_block'][1].start, params[0]['contextless_slices_wrt_block'][1].stop)
                                            ] for slice_index in range(params[0]['contextless_slices_wrt_block'][0].start, params[0]['contextless_slices_wrt_block'][0].stop)
                                        ], np.int32),
                                    params[0]['contextless_slices_wrt_whole']+(slice(None),)
                                ),
                        block_shape=[4]*3, context_size=1, n_jobs=1
                    ),
                np.array([
                        [
                            [
                                np.histogram(regions.get_neighbourhood_array_3d(scaled_data[0], (slice_index, row, col), 1, {0,1,2}), 7*7*7, (1, 7*7*7+1))[0]
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ] for slice_index in range(scaled_data[0].shape[0])
                    ], np.int32)
            )
    
    #########################################
    def test_process_array_in_blocks_single_slice(self):
        scaled_data = { 0: (np.arange(5*5*5) + 1).reshape([5,5,5]) }
        
        for slice_index in range(scaled_data[0].shape[0]):
            out = np.zeros([5,5], scaled_data[0].dtype)
            
            with self.assertRaises(ValueError):
                out[:,:] = 0
                arrayprocs.process_array_in_blocks_single_slice(
                    scaled_data, out,
                    lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                    block_shape=[2]*2, slice_index=slice_index, context_size=1, n_jobs=1
                )
            with self.assertRaises(ValueError):
                out[:,:] = 0
                arrayprocs.process_array_in_blocks_single_slice(
                    scaled_data, out,
                    lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                    block_shape=[2]*2, slice_index=slice_index, context_size=2, n_jobs=1
                )
            with self.assertRaises(ValueError):
                out[:,:] = 0
                arrayprocs.process_array_in_blocks_single_slice(
                    scaled_data, out,
                    lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                    block_shape=[4]*2, slice_index=slice_index, context_size=2, n_jobs=1
                )
            
            out[:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                            block_shape=[1]*2, slice_index=slice_index, context_size=0, n_jobs=1
                        ),
                    scaled_data[0][slice_index,:,:]+1,
                    'slice_index={}'.format(slice_index)
                )
            
            out[:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                            block_shape=[2]*2, slice_index=slice_index, context_size=0, n_jobs=1
                        ),
                    scaled_data[0][slice_index,:,:]+1,
                    'slice_index={}'.format(slice_index)
                )
            
            out[:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                            block_shape=[3]*2, slice_index=slice_index, context_size=1, n_jobs=1
                        ),
                    scaled_data[0][slice_index,:,:]+1,
                    'slice_index={}'.format(slice_index)
                )
            
            out[:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                            block_shape=[6]*2, slice_index=slice_index, context_size=2, n_jobs=1
                        ),
                    scaled_data[0][slice_index,:,:]+1,
                    'slice_index={}'.format(slice_index)
                )
            
            out[:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(params[0]['block'][params[0]['contextless_slices_wrt_block']]+1, params[0]['contextless_slices_wrt_whole'][1:]),
                            block_shape=[6]*2, slice_index=slice_index, context_size=2, n_jobs=2
                        ),
                    scaled_data[0][slice_index,:,:]+1,
                    'slice_index={}'.format(slice_index)
                )
            
            out = np.zeros([5,5,5*5*5], np.int32)
        
            out[:,:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(np.histogram(params[0]['block'], 5*5*5, (1, 5*5*5+1))[0], params[0]['contextless_slices_wrt_whole'][1:]+(slice(None),)),
                            block_shape=[3]*3, slice_index=slice_index, context_size=1, n_jobs=1
                        ),
                    np.array([
                            [
                                np.histogram(regions.get_neighbourhood_array_3d(scaled_data[0], (slice_index, row, col), 1, {0,1,2}), 5*5*5, (1, 5*5*5+1))[0]
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ], np.int32),
                    'slice_index={}'.format(slice_index)
                )
            
            out[:,:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:(np.histogram(params[0]['block'], 5*5*5, (1, 5*5*5+1))[0], params[0]['contextless_slices_wrt_whole'][1:]+(slice(None),)),
                            block_shape=[5]*3, slice_index=slice_index, context_size=2, n_jobs=1
                        ),
                    np.array([
                            [
                                np.histogram(regions.get_neighbourhood_array_3d(scaled_data[0], (slice_index, row, col), 2, {0,1,2}), 5*5*5, (1, 5*5*5+1))[0]
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ], np.int32),
                    'slice_index={}'.format(slice_index)
                )
            
            out[:,:,:] = 0
            np.testing.assert_equal(
                    arrayprocs.process_array_in_blocks_single_slice(
                            scaled_data, out,
                            lambda params:
                                    (
                                        np.array([
                                                [
                                                    np.histogram(regions.get_neighbourhood_array_3d(params[0]['block'], (params[0]['contextless_slices_wrt_block'][0], row, col), 1, {0,1,2}), 5*5*5, (1, 5*5*5+1))[0]
                                                    for col in range(params[0]['contextless_slices_wrt_block'][2].start, params[0]['contextless_slices_wrt_block'][2].stop)
                                                ] for row in range(params[0]['contextless_slices_wrt_block'][1].start, params[0]['contextless_slices_wrt_block'][1].stop)
                                            ], np.int32),
                                        params[0]['contextless_slices_wrt_whole'][1:]+(slice(None),)
                                    ),
                            block_shape=[4]*3, slice_index=slice_index, context_size=1, n_jobs=1
                        ),
                    np.array([
                            [
                                np.histogram(regions.get_neighbourhood_array_3d(scaled_data[0], (slice_index, row, col), 1, {0,1,2}), 5*5*5, (1, 5*5*5+1))[0]
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ], np.int32),
                    'slice_index={}'.format(slice_index)
                )

if __name__ == '__main__':
    unittest.main()
