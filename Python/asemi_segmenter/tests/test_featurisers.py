import unittest
import numpy as np
import os
import sys
import skimage.feature
from asemi_segmenter.lib import featurisers
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import regions
from asemi_segmenter.lib import histograms

#########################################
class Features(unittest.TestCase):

    #########################################
    def test_histogram_featuriser_voxels(self):
        indexes = [(0,0,0), (2,7,7), (4,14,14)]
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (radius, scale, num_bins) in [
                    (2, 0, 16),
                    (1, 1, 16),
                    (1, 0, 16),
                ]:
                featuriser = featurisers.HistogramFeaturiser(radius, scale, num_bins)
                true_features = []
                for index in indexes:
                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], index, radius, {0,1,2}, scale=scale)
                    true_features.append(histograms.histogram(neighbourhood, num_bins, (0, 2**16)))
                true_features = np.array(true_features, np.float32)

                features = featuriser.featurise_voxels(scaled_data, indexes)
                np.testing.assert_equal(true_features, features, 'downsample_kernel={}, featuriser_params={}'.format(downsample_kernel.name, (radius, scale, num_bins)))

    #########################################
    def test_histogram_featuriser_slice(self):
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
                featuriser = featurisers.HistogramFeaturiser(radius, scale, num_bins)
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

                    slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, n_jobs=n_jobs)
                    np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, featuriser_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, (radius, scale, num_bins), batch_size, n_jobs, slice_index))

                    output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
                    expected_output = np.zeros_like(output)
                    expected_output[2:-2, :] = slice_features
                    slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2, n_jobs=n_jobs)
                    np.testing.assert_equal(expected_output, slice_features, 'downsample_kernel={}, featuriser_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, (radius, scale, num_bins), batch_size, n_jobs, slice_index))

    #########################################
    def test_lbp_featuriser_voxels(self):
        indexes = [(0,0,0), (2,7,7), (4,14,14)]
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (neighbouring_dims, radius, scale) in [
                    ({0,1}, 2, 0),
                    ({0,1}, 1, 1),
                    ({0,1}, 1, 0),
                    ({0,2}, 2, 0),
                    ({0,2}, 1, 1),
                    ({0,2}, 1, 0),
                    ({1,2}, 2, 0),
                    ({1,2}, 1, 1),
                    ({1,2}, 1, 0)
                ]:
                featuriser = featurisers.LocalBinaryPatternFeaturiser(neighbouring_dims, radius, scale)
                true_features = []
                for index in indexes:
                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], index, radius+1, neighbouring_dims, scale=scale)
                    lbp = skimage.feature.local_binary_pattern(neighbourhood, 8, 1, 'uniform')[1:-1,1:-1]
                    true_features.append(histograms.histogram(lbp, 10, (0, 10)))
                true_features = np.array(true_features, np.float32)

                features = featuriser.featurise_voxels(scaled_data, indexes)
                np.testing.assert_equal(true_features, features, 'downsample_kernel={}, featuriser_params={}'.format(downsample_kernel.name, (neighbouring_dims, radius, scale)))

    #########################################
    def test_lbp_featuriser_slice(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            #scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,3,3), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (neighbouring_dims, radius, scale, batch_size, n_jobs) in [
                    ({0,1}, 2, 0, 12, 1),
                    ({0,1}, 1, 1, 12, 1),
                    ({0,1}, 2, 0, 200, 1),
                    ({0,1}, 1, 1, 200, 1),
                    ({0,1}, 1, 0, 10, 2),
                    ({0,2}, 2, 0, 12, 1),
                    ({0,2}, 1, 1, 12, 1),
                    ({0,2}, 2, 0, 200, 1),
                    ({0,2}, 1, 1, 200, 1),
                    ({1,2}, 1, 0, 10, 2),
                    ({1,2}, 2, 0, 12, 1),
                    ({1,2}, 1, 1, 12, 1),
                    ({1,2}, 2, 0, 200, 1),
                    ({1,2}, 1, 1, 200, 1),
                    ({1,2}, 1, 0, 10, 2),
                ]:
                featuriser = featurisers.LocalBinaryPatternFeaturiser(neighbouring_dims, radius, scale)
                for slice_index in range(scaled_data[0].shape[0]):
                    true_slice_features = [
                            [
                                []
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ]
                    for row in range(scaled_data[0].shape[1]):
                        for col in range(scaled_data[0].shape[2]):
                            neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], (slice_index, row, col), radius+1, neighbouring_dims, scale=scale)
                            lbp = skimage.feature.local_binary_pattern(neighbourhood, 8, 1, 'uniform')[1:-1,1:-1]
                            true_slice_features[row][col].extend(histograms.histogram(lbp, 10, (0, 10)))
                    true_slice_features = np.array(true_slice_features, np.float32).reshape([-1, len(true_slice_features[0][0])])

                    slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, n_jobs=n_jobs)
                    np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, featuriser_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, (neighbouring_dims, radius, scale), batch_size, n_jobs, slice_index))

                    output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
                    expected_output = np.zeros_like(output)
                    expected_output[2:-2, :] = slice_features
                    slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2, n_jobs=n_jobs)
                    np.testing.assert_equal(expected_output, slice_features, 'downsample_kernel={}, featuriser_params={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, (neighbouring_dims, radius, scale), batch_size, n_jobs, slice_index))

    #########################################
    def test_voxel_featuriser_voxels(self):
        indexes = [(0,0,0), (2,7,7), (4,14,14)]
        rand = np.random.RandomState(0)
        scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }

        featuriser = featurisers.VoxelFeaturiser()
        true_features = []
        for index in indexes:
            true_features.append([scaled_data[0][index]])
        true_features = np.array(true_features, np.float32)

        features = featuriser.featurise_voxels(scaled_data, indexes)
        np.testing.assert_equal(true_features, features)

    #########################################
    def test_voxel_featuriser_slice(self):
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

            slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size)
            np.testing.assert_equal(true_slice_features, slice_features, 'slice_index={}'.format(slice_index))

            output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
            expected_output = np.zeros_like(output)
            expected_output[2:-2, :] = slice_features
            slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2)
            np.testing.assert_equal(expected_output, slice_features, 'slice_index={}'.format(slice_index))

    #########################################
    def test_composite_featuriser_voxels(self):
        indexes = [(0,0,0), (2,7,7), (4,14,14)]
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (name, featuriser_list) in [
                    ('v-h', [ featurisers.VoxelFeaturiser(), featurisers.HistogramFeaturiser(2, 0, 16) ]),
                ]:
                featuriser = featurisers.CompositeFeaturiser(featuriser_list)
                true_features = []
                for index in indexes:
                    feature_vec = []
                    for sub_featuriser in featuriser_list:
                        if isinstance(sub_featuriser, featurisers.VoxelFeaturiser):
                            feature_vec.append(scaled_data[0][index])
                        elif isinstance(sub_featuriser, featurisers.HistogramFeaturiser):
                            neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[sub_featuriser.scale], index, sub_featuriser.radius, {0,1,2}, scale=sub_featuriser.scale)
                            feature_vec.extend(histograms.histogram(neighbourhood, sub_featuriser.num_bins, (0, 2**16)))
                    true_features.append(feature_vec)
                true_features = np.array(true_features, np.float32)

                features = featuriser.featurise_voxels(scaled_data, indexes)
                np.testing.assert_equal(true_features, features, 'downsample_kernel={}, name={}'.format(downsample_kernel.name, name))

    #########################################
    def test_composite_featuriser_slice(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (name, featuriser_list, batch_size, n_jobs) in [
                    ('v-h', [ featurisers.VoxelFeaturiser(), featurisers.HistogramFeaturiser(2, 0, 16) ], 12, 1),
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
                                elif isinstance(sub_featuriser, featurisers.HistogramFeaturiser):
                                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[sub_featuriser.scale], (slice_index, row, col), sub_featuriser.radius, {0,1,2}, scale=sub_featuriser.scale)
                                    true_slice_features[row][col].extend(histograms.histogram(neighbourhood, sub_featuriser.num_bins, (0, 2**16)))
                    true_slice_features = np.array(true_slice_features, np.float32).reshape([-1, len(true_slice_features[0][0])])

                    slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, n_jobs=n_jobs)
                    np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, name={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, name, batch_size, n_jobs, slice_index))

                    output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1] ], np.float32)
                    expected_output = np.zeros_like(output)
                    expected_output[2:-2, :] = slice_features
                    slice_features = featuriser.featurise_slice(scaled_data, slice_index, block_rows=batch_size, block_cols=batch_size, output=output, output_start_row_index=2, n_jobs=n_jobs)
                    np.testing.assert_equal(expected_output, slice_features, 'downsample_kernel={}, name={}, batch_size={}, n_jobs={}, slice_index={}'.format(downsample_kernel.name, name, batch_size, n_jobs, slice_index))

if __name__ == '__main__':
    unittest.main()
