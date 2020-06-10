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
    def test_get_optimal_block_size(self):
        #Check that tiny max memory results in error.
        '''
        with self.assertRaises(ValueError):
            arrayprocs.get_optimal_block_size(data_shape=(1000,2000,3000), data_dtype=np.uint16, context_needed=1, num_processes=1, max_batch_memory_gb=(1*2)/(1024**3), num_implicit_slices=None, feature_size=None, feature_dtype=None)
        with self.assertRaises(ValueError):
            arrayprocs.get_optimal_block_size(data_shape=(1000,2000,3000), data_dtype=np.uint16, context_needed=1, num_processes=1, max_batch_memory_gb=(1*11*4)/(1024**3), num_implicit_slices=1, feature_size=11, feature_dtype=np.float32)
        '''
        for params in [
            dict(data_shape=(1000,2000,3000), data_dtype=np.uint16, context_needed=10, num_processes=1, max_elements=500000, num_implicit_slices=None, feature_size=None, feature_dtype=None),
            dict(data_shape=(1000,2000,3000), data_dtype=np.uint16, context_needed=10, num_processes=5, max_elements=5*500000, num_implicit_slices=None, feature_size=None, feature_dtype=None),
            dict(data_shape=(101,201), data_dtype=np.uint16, context_needed=1, num_processes=1, max_elements=1000000, num_implicit_slices=1, feature_size=1, feature_dtype=np.float32),
            dict(data_shape=(1000,2000), data_dtype=np.uint16, context_needed=10, num_processes=1, max_elements=10000000, num_implicit_slices=2, feature_size=11, feature_dtype=np.float32),
            dict(data_shape=(1000,2000), data_dtype=np.uint16, context_needed=10, num_processes=5, max_elements=5*10000000, num_implicit_slices=2, feature_size=11, feature_dtype=np.float32),
            ]:
            (data_shape, data_dtype, context_needed, num_processes, max_elements, num_implicit_slices, feature_size, feature_dtype) = (params['data_shape'], params['data_dtype'], params['context_needed'], params['num_processes'], params['max_elements'], params['num_implicit_slices'], params['feature_size'], params['feature_dtype'])

            max_batch_memory_gb = max_elements*np.dtype(data_dtype).itemsize/(1024**3)

            params_str = 'data_shape={}, data_dtype={}, context_needed={}, num_processes={}, max_elements={}, num_implicit_slices={}, feature_size={}, feature_dtype={}'.format(data_shape, data_dtype, context_needed, num_processes, max_elements, num_implicit_slices, feature_size, feature_dtype)

            block_shape = arrayprocs.get_optimal_block_size(data_shape, data_dtype, context_needed, num_processes, max_batch_memory_gb, num_implicit_slices, feature_size, feature_dtype)

            if num_implicit_slices is None:
                self.assertEqual(len(block_shape), 3, params_str)
            else:
                self.assertEqual(len(block_shape), 2, params_str)

            data_space = np.prod(block_shape).tolist()*np.dtype(data_dtype).itemsize/(1024**3)
            if num_implicit_slices is not None:
                data_space *= 2*context_needed + num_implicit_slices
            self.assertLessEqual(data_space, max_batch_memory_gb, params_str)

            if feature_size is not None:
                features_space = np.prod([l-2*context_needed for l in block_shape]).tolist()*feature_size*np.dtype(feature_dtype).itemsize/(1024**3)
                if num_implicit_slices is not None:
                    features_space *= num_implicit_slices
                self.assertLessEqual(features_space, max_batch_memory_gb, params_str)

            for i in range(len(block_shape)):
                if block_shape[i] - 2*context_needed != data_shape[i]:
                    self.assertEqual((block_shape[i] - 2*context_needed)%32, 0, params_str)

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
    def test_process_array_in_blocks_slice_range(self):
        scaled_data = { 0: (np.arange(5*5*5) + 1).reshape([5,5,5]) }

        for num_slices in range(1, 3):
            for slice_index in range(scaled_data[0].shape[0]):
                actual_num_slices = min(scaled_data[0].shape[0] - slice_index, num_slices)
                out = np.zeros([actual_num_slices, 5, 5], scaled_data[0].dtype)
                slice_range = slice(slice_index, slice_index+num_slices)

                def processor(params):
                    return (
                        params[0]['block'][params[0]['contextless_slices_wrt_block']] + 1,
                        params[0]['contextless_slices_wrt_range']
                        )
                expected_output = scaled_data[0][slice_range,:,:] + 1

                out[:, :, :] = 0
                np.testing.assert_equal(
                        arrayprocs.process_array_in_blocks_slice_range(
                                scaled_data, out,
                                processor,
                                block_shape=[1]*2, slice_range=slice_range, context_size=0, n_jobs=1
                            ),
                        expected_output,
                        'slice_range={}'.format(slice_range)
                    )

                out[:, :, :] = 0
                np.testing.assert_equal(
                        arrayprocs.process_array_in_blocks_slice_range(
                                scaled_data, out,
                                processor,
                                block_shape=[2]*2, slice_range=slice_range, context_size=0, n_jobs=1
                            ),
                        expected_output,
                        'slice_range={}'.format(slice_range)
                    )

                out[:, :, :] = 0
                np.testing.assert_equal(
                        arrayprocs.process_array_in_blocks_slice_range(
                                scaled_data, out,
                                processor,
                                block_shape=[3]*2, slice_range=slice_range, context_size=1, n_jobs=1
                            ),
                        expected_output,
                        'slice_range={}'.format(slice_range)
                    )

                out[:, :, :] = 0
                np.testing.assert_equal(
                        arrayprocs.process_array_in_blocks_slice_range(
                                scaled_data, out,
                                processor,
                                block_shape=[6]*2, slice_range=slice_range, context_size=2, n_jobs=1
                            ),
                        expected_output,
                        'slice_range={}'.format(slice_range)
                    )

                out[:, :, :] = 0
                np.testing.assert_equal(
                        arrayprocs.process_array_in_blocks_slice_range(
                                scaled_data, out,
                                processor,
                                block_shape=[6]*2, slice_range=slice_range, context_size=2, n_jobs=2
                            ),
                        expected_output,
                        'slice_range={}'.format(slice_range)
                    )

                out = np.zeros([actual_num_slices,5,5,3,3,3], scaled_data[0].dtype)
                expected_output = np.array([
                        [
                            [
                                regions.get_neighbourhood_array_3d(scaled_data[0], (slc, row, col), 1, {0,1,2})
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ] for slc in range(scaled_data[0].shape[0])
                    ], scaled_data[0].dtype)[slice_range,:,:,:]

                out[:,:,:,:,:,:] = 0
                np.testing.assert_equal(
                        arrayprocs.process_array_in_blocks_slice_range(
                                scaled_data, out,
                                lambda params: (
                                        np.array([
                                                [
                                                    [
                                                        regions.get_neighbourhood_array_3d(params[0]['block'], (slc, row, col), 1, {0,1,2})
                                                        for col in range(params[0]['contextless_slices_wrt_block'][2].start, params[0]['contextless_slices_wrt_block'][2].stop)
                                                    ] for row in range(params[0]['contextless_slices_wrt_block'][1].start, params[0]['contextless_slices_wrt_block'][1].stop)
                                                ] for slc in range(params[0]['contextless_slices_wrt_block'][0].start, params[0]['contextless_slices_wrt_block'][0].stop)
                                            ], scaled_data[0].dtype),
                                        params[0]['contextless_slices_wrt_range']
                                    ),
                                block_shape=[4]*2, slice_range=slice_range, context_size=1, n_jobs=1
                            ),
                            expected_output,
                            'slice_range={}'.format(slice_range)
                        )


if __name__ == '__main__':
    unittest.main()
