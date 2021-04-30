#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 Marc Tanti
#
# This file is part of ASEMI-segmenter.
#
# ASEMI-segmenter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ASEMI-segmenter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASEMI-segmenter.  If not, see <http://www.gnu.org/licenses/>.

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
class Featurisers(unittest.TestCase):

    #########################################
    def test_voxel_featuriser_voxels(self):
        indexes = [(0,0,0), (2,7,7), (4,14,14)]
        rand = np.random.RandomState(0)
        scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }

        true_features = []
        for index in indexes:
            true_features.append([scaled_data[0][index]])
        true_features = np.array(true_features, featurisers.feature_dtype)

        featuriser = featurisers.VoxelFeaturiser()
        features = featuriser.featurise_voxels(scaled_data, indexes)

        np.testing.assert_equal(true_features, features)

    #########################################
    def test_voxel_featuriser_slice(self):
        rand = np.random.RandomState(0)
        scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
        batch_size = None

        true_features = [
                [
                    [
                        []
                        for col in range(scaled_data[0].shape[2])
                    ] for row in range(scaled_data[0].shape[1])
                ] for slc in range(scaled_data[0].shape[0])
            ]
        for slc in range(scaled_data[0].shape[0]):
            for row in range(scaled_data[0].shape[1]):
                for col in range(scaled_data[0].shape[2]):
                    true_features[slc][row][col].append(scaled_data[0][slc,row,col])
        true_features = np.array(true_features, featurisers.feature_dtype)

        featuriser = featurisers.VoxelFeaturiser()
        for slice_index in range(scaled_data[0].shape[0]):
            for num_slices in range(1, 3+1):
                slice_range = slice(slice_index, slice_index+num_slices)

                true_slice_features = true_features[slice_range,:,:].reshape([-1, true_features.shape[-1]])
                slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size))
                np.testing.assert_equal(true_slice_features, slice_features, 'slice_range={}'.format(slice_range))

                output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1]+4 ], featurisers.feature_dtype)
                expected_output = np.zeros_like(output)
                expected_output[2:-2, 2:-2] = slice_features
                slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size), output=output, output_start_row_index=2, output_start_col_index=2)
                np.testing.assert_equal(expected_output, slice_features, 'slice_range={}'.format(slice_range))

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
                true_features = []
                for index in indexes:
                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], index, radius, {0,1,2}, scale=scale)
                    true_features.append(histograms.histogram(neighbourhood, num_bins, (0, 2**16)))
                true_features = np.array(true_features, featurisers.feature_dtype)

                featuriser = featurisers.HistogramFeaturiser(radius, scale, num_bins)
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

            for (radius, scale, num_bins, batch_size, max_processes) in [
                    (2, 0, 16, 12, 1),
                    (1, 1, 16, 12, 1),
                    (2, 0, 16, 200, 1),
                    (1, 1, 16, 200, 1),
                    (1, 0, 16, 10, 2),
                ]:
                true_features = [
                        [
                            [
                                []
                                for col in range(scaled_data[scale].shape[2])
                            ] for row in range(scaled_data[scale].shape[1])
                        ] for slc in range(scaled_data[scale].shape[0])
                    ]
                for slc in range(scaled_data[scale].shape[0]):
                    for row in range(scaled_data[scale].shape[1]):
                        for col in range(scaled_data[scale].shape[2]):
                            neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], (slc, row, col), radius, {0,1,2})
                            hist = histograms.histogram(neighbourhood, num_bins, (0, 2**16))
                            true_features[slc][row][col].extend(hist)
                true_features = np.array(true_features, featurisers.feature_dtype)
                true_features = downscales.grow_array(true_features, scale, [0,1,2], scaled_data[0].shape)

                featuriser = featurisers.HistogramFeaturiser(radius, scale, num_bins)

                for slice_index in range(scaled_data[0].shape[0]):
                    for num_slices in range(1, 3):
                        slice_range = slice(slice_index, slice_index+num_slices)

                        true_slice_features = true_features[slice_range,:,:].reshape([-1, true_features.shape[-1]])
                        slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size), max_processes=max_processes)
                        np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, featuriser_params={}, batch_size={}, max_processes={}, slice_range={}'.format(downsample_kernel.name, (radius, scale, num_bins), batch_size, max_processes, slice_range))

                        output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1]+4 ], featurisers.feature_dtype)
                        expected_output = np.zeros_like(output)
                        expected_output[2:-2, 2:-2] = slice_features
                        slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size), output=output, output_start_row_index=2, output_start_col_index=2)
                        np.testing.assert_equal(expected_output, slice_features, 'slice_range={}'.format(slice_range))

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
                true_features = []
                for index in indexes:
                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], index, radius+1, neighbouring_dims, scale=scale)
                    lbp = skimage.feature.local_binary_pattern(neighbourhood, 8, 1, 'uniform')[1:-1,1:-1]
                    true_features.append(histograms.histogram(lbp, 10, (0, 10)))
                true_features = np.array(true_features, featurisers.feature_dtype)

                featuriser = featurisers.LocalBinaryPatternFeaturiser(neighbouring_dims, radius, scale)
                features = featuriser.featurise_voxels(scaled_data, indexes)

                np.testing.assert_equal(true_features, features, 'downsample_kernel={}, featuriser_params={}'.format(downsample_kernel.name, (neighbouring_dims, radius, scale)))

    #########################################
    def test_lbp_featuriser_slice(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            rand = np.random.RandomState(0)
            scaled_data = { 0: rand.randint(0, 2**16-1, (5,15,15), np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downsample_kernel, 1)
            for (neighbouring_dims, radius, scale, batch_size, max_processes) in [
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
                true_features = [
                        [
                            [
                                []
                                for col in range(scaled_data[scale].shape[2])
                            ] for row in range(scaled_data[scale].shape[1])
                        ] for slc in range(scaled_data[scale].shape[0])
                    ]
                for slc in range(scaled_data[scale].shape[0]):
                    for row in range(scaled_data[scale].shape[1]):
                        for col in range(scaled_data[scale].shape[2]):
                            neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[scale], (slc, row, col), radius+1, neighbouring_dims)
                            lbp = skimage.feature.local_binary_pattern(neighbourhood, 8, 1, 'uniform')[1:-1,1:-1]
                            hist = histograms.histogram(lbp, 10, (0, 10))
                            true_features[slc][row][col].extend(hist)
                true_features = np.array(true_features, featurisers.feature_dtype)
                true_features = downscales.grow_array(true_features, scale, [0,1,2], scaled_data[0].shape)

                featuriser = featurisers.LocalBinaryPatternFeaturiser(neighbouring_dims, radius, scale)
                for slice_index in range(scaled_data[0].shape[0]):
                    for num_slices in range(1, 3):
                        slice_range = slice(slice_index, slice_index+num_slices)

                        true_slice_features = true_features[slice_range,:,:].reshape([-1, true_features.shape[-1]])
                        slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size), max_processes=max_processes)
                        np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, featuriser_params={}, batch_size={}, max_processes={}, slice_range={}'.format(downsample_kernel.name, (radius, scale), batch_size, max_processes, slice_range))

                        output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1]+4 ], featurisers.feature_dtype)
                        expected_output = np.zeros_like(output)
                        expected_output[2:-2, 2:-2] = slice_features
                        slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size), output=output, output_start_row_index=2, output_start_col_index=2)
                        np.testing.assert_equal(expected_output, slice_features, 'slice_range={}'.format(slice_range))

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
                true_features = np.array(true_features, featurisers.feature_dtype)

                featuriser = featurisers.CompositeFeaturiser(featuriser_list)
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
            for (name, featuriser_list, batch_size, max_processes) in [
                    ('v-h', [ featurisers.VoxelFeaturiser(), featurisers.HistogramFeaturiser(2, 0, 16) ], 12, 1),
                ]:
                true_features = [
                        [
                            [
                                []
                                for col in range(scaled_data[0].shape[2])
                            ] for row in range(scaled_data[0].shape[1])
                        ] for slc in range(scaled_data[0].shape[0])
                    ]
                for slc in range(scaled_data[0].shape[0]):
                    for row in range(scaled_data[0].shape[1]):
                        for col in range(scaled_data[0].shape[2]):
                            for sub_featuriser in featuriser_list:
                                if isinstance(sub_featuriser, featurisers.VoxelFeaturiser):
                                    true_features[slc][row][col].append(scaled_data[0][slc, row, col])
                                elif isinstance(sub_featuriser, featurisers.HistogramFeaturiser):
                                    neighbourhood = regions.get_neighbourhood_array_3d(scaled_data[sub_featuriser.scale], (slc, row, col), sub_featuriser.radius, {0,1,2}, scale=sub_featuriser.scale)
                                    true_features[slc][row][col].extend(histograms.histogram(neighbourhood, sub_featuriser.num_bins, (0, 2**16)))
                true_features = np.array(true_features, featurisers.feature_dtype)

                featuriser = featurisers.CompositeFeaturiser(featuriser_list)
                for slice_index in range(scaled_data[0].shape[0]):
                    for num_slices in range(1, 3):
                        slice_range = slice(slice_index, slice_index+num_slices)

                        true_slice_features = true_features[slice_range,:,:].reshape([-1, true_features.shape[-1]])
                        slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size))
                        np.testing.assert_equal(true_slice_features, slice_features, 'downsample_kernel={}, name={}, batch_size={}, max_processes={}, slice_range={}'.format(downsample_kernel.name, name, batch_size, max_processes, slice_range))

                        output = np.zeros([ slice_features.shape[0]+4, slice_features.shape[1]+4 ], featurisers.feature_dtype)
                        expected_output = np.zeros_like(output)
                        expected_output[2:-2, 2:-2] = slice_features
                        slice_features = featuriser.featurise_slice(scaled_data, slice_range, block_shape=(batch_size, batch_size), output=output, output_start_row_index=2, output_start_col_index=2)
                        np.testing.assert_equal(expected_output, slice_features, 'slice_range={}'.format(slice_range))


if __name__ == '__main__':
    unittest.main()
