import unittest
import numpy as np
import os
import sys
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import histograms

#########################################
class Features(unittest.TestCase):
    
    #########################################
    def test_histogram_featuriser2(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (radius, scale, num_bins, batch_size, n_jobs) in [
                    (2, 0, 16, 12, 1),
                    (1, 1, 16, 12, 1),
                    (2, 0, 16, 200, 1),
                    (1, 1, 16, 200, 1),
                    (1, 0, 16, 10, 2),
                ]:
                featuriser = featurisers.HistogramFeaturiser2(radius, scale, num_bins)
                for slice_index in range(scaled_data[0].shape[0]):
                    true_slice_features = [
                            [
                                []
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ]
                    for row in range(scaled_data[0].shape[1]):
                        for col in range(scaled_data[0].shape[2]):
                            neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], (slice_index, row, col), radius, {0,1,2}, scale=scale)
                            true_slice_features[row][col].extend(histograms.histogram(neighbourhood, num_bins, (0, 2**16)))
                    true_slice_features = np.array(true_slice_features, np.float32).reshape([-1, len(true_slice_features[0][0])])
                    
                    slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, n_jobs=n_jobs)
                    np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, histogram_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, (radius, scale, num_bins), batch_size, n_jobs, slice_index))
                    
                    output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
                    expected_output = np.zeros_like(output)
                    expected_output[2:-2, :] = slice_features
                    slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2, n_jobs=n_jobs)
                    np.testing.assert_equal(expected_output, slice_features, 'downsample_kernel={}, histogram_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, (radius, scale, num_bins), batch_size, n_jobs, slice_index))

    #########################################
    def test_voxel_featuriser(self):
        rand = np.random.RandomState(0)
        scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
        batch_size = None
        featuriser = featurisers.VoxelFeaturiser()
        for slice_index in range(scaled_data[0].shape[0]):
            true_slice_features = [
                    [
                        []
                        for col in range(scaled_data[0].shape[2])
                    ] for row in range(scaled_data[0].shape[1])
                ]
            for row in range(scaled_data[0].shape[1]):
                for col in range(scaled_data[0].shape[2]):
                    true_slice_features[row][col].append(scaled_data[0][slice_index, row, col])
            true_slice_features = np.array(true_slice_features, np.float32).reshape([-1, len(true_slice_features[0][0])])
            
            slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size)
            np.testing.assert_equal(true_slice_features, slice_features, 'slice_index={}'.format(slice_index))
            
            output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
            expected_output = np.zeros_like(output)
            expected_output[2:-2, :] = slice_features
            slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2)
            np.testing.assert_equal(expected_output, slice_features, 'slice_index={}'.format(slice_index))

    #########################################
    def test_composite_featuriser(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (name, featuriser_list, batch_size, n_jobs) in [
                    ('v-h', [ featurisers.VoxelFeaturiser(), featurisers.HistogramFeaturiser2(2, 0, 16) ], 12, 1),
                ]:
                featuriser = featurisers.CompositeFeaturiser(featuriser_list)
                for slice_index in range(scaled_data[0].shape[0]):
                    true_slice_features = [
                            [
                                []
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ]
                    for row in range(scaled_data[0].shape[1]):
                        for col in range(scaled_data[0].shape[2]):
                            for sub_featuriser in featuriser_list:
                                if isinstance(sub_featuriser, featurisers.VoxelFeaturiser):
                                    true_slice_features[row][col].append(scaled_data[0][slice_index, row, col])
                                elif isinstance(sub_featuriser, featurisers.HistogramFeaturiser2):
                                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[sub_featuriser.scale], (slice_index, row, col), sub_featuriser.radius, {0,1,2}, scale=sub_featuriser.scale)
                                    true_slice_features[row][col].extend(histograms.histogram(neighbourhood, sub_featuriser.num_bins, (0, 2**16)))
                    true_slice_features = np.array(true_slice_features, np.float32).reshape([-1, len(true_slice_features[0][0])])
                    
                    slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, n_jobs=n_jobs)
                    np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, name={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, name, batch_size, n_jobs, slice_index))
                    
                    output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
                    expected_output = np.zeros_like(output)
                    expected_output[2:-2, :] = slice_features
                    slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2, n_jobs=n_jobs)
                    np.testing.assert_equal(expected_output, slice_features, 'downsample_kernel={}, name={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, name, batch_size, n_jobs, slice_index))

    #########################################
    def test_histogram_featuriser(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (use_voxel_value, histogram_params, batch_size, n_jobs) in [
                    (True, [(1, 0, 16), (2, 0, 16)], 12, 1),
                    (True, [(1, 0, 16), (1, 1, 16)], 12, 1),
                    (True, [(1, 0, 16), (2, 0, 16)], 200, 1),
                    (True, [(1, 0, 16), (1, 1, 16)], 200, 1),
                    (True, [(1, 0, 16)], 10, 2),
                ]:
                for slice_index in range(scaled_data[0].shape[0]):
                    true_slice_features = [
                            [
                                []
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ]
                    for row in range(scaled_data[0].shape[1]):
                        for col in range(scaled_data[0].shape[2]):
                            if use_voxel_value:
                                true_slice_features[row][col].append(scaled_data[0][slice_index, row, col])
                            for (radius, scale, num_histogram_bins) in histogram_params:
                                neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], (slice_index, row, col), radius, {0,1,2}, scale=scale)
                                true_slice_features[row][col].extend(histograms.histogram(neighbourhood, num_histogram_bins, (0, 2**16)))
                    true_slice_features = np.array(true_slice_features, np.float32).reshape([-1, len(true_slice_features[0][0])])
                    
                    featuriser = featurisers.HistogramFeaturiser(use_voxel_value, histogram_params)
                    
                    slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, n_jobs=n_jobs)
                    np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, histogram_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, histogram_params, batch_size, n_jobs, slice_index))
                    
                    output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
                    expected_output = np.zeros_like(output)
                    expected_output[2:-2, :] = slice_features
                    slice_features = featuriser.featurise(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2, n_jobs=n_jobs)
                    np.testing.assert_equal(expected_output, slice_features, 'downsample_kernel={}, histogram_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, histogram_params, batch_size, n_jobs, slice_index))
                    
if __name__ == '__main__':
    unittest.main()
