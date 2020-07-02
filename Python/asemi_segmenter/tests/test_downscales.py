import unittest
import numpy as np
import os
import sys
from asemi_segmenter.lib import downscales

#########################################
class Downscales(unittest.TestCase):

    #########################################
    def test_grow_array(self):
        np.testing.assert_equal(
                downscales.grow_array(np.array([ 0, 1, 2 ]), 0, [0]),
                np.array([ 0, 1, 2 ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ 0, 1, 2 ]), 1, [0]),
                np.array([ 0, 0, 1, 1, 2, 2 ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ 0, 1, 2 ]), 2, [0]),
                np.array([ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ 0, 1, 2 ]), 0, [0], orig_shape=[ 5 ]),
                np.array([ 0, 1, 2 ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ 0, 1, 2 ]), 1, [0], orig_shape=[ 5 ]),
                np.array([ 0, 0, 1, 1, 2 ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ 0, 1, 2 ]), 2, [0], orig_shape=[ 5 ]),
                np.array([ 0, 0, 0, 0, 1 ])
            )


        np.testing.assert_equal(
                downscales.grow_array(np.array([ [ 0, 1 ], [ 2, 3 ] ]), 1, [0]),
                np.array([ [ 0, 1 ], [ 0, 1 ], [ 2, 3 ], [ 2, 3 ] ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ [ 0, 1 ], [ 2, 3 ] ]), 1, [1]),
                np.array([ [ 0, 0, 1, 1 ], [ 2, 2, 3, 3 ] ])
            )

        np.testing.assert_equal(
                downscales.grow_array(np.array([ [ 0, 1 ],[ 2, 3 ] ]), 1, [0, 1]),
                np.array([ [ 0, 0, 1, 1 ], [ 0, 0, 1, 1 ], [ 2, 2, 3, 3 ], [ 2, 2, 3, 3 ] ])
            )

    #########################################
    def test_partial_growing(self):
        orig_array = np.arange(2**4)
        for scale in range(3):
            reduced = downscales.downscale(orig_array, downscales.NullDownsampleKernel(), scale)
            grown = downscales.grow_array(reduced, scale, [0], orig_array.shape)
            for index1 in range(0, orig_array.shape[0]-1):
                for index2 in range(index1+1, orig_array.shape[0]):
                    part_range = slice(index1, index2)
                    part_size = part_range.stop - part_range.start

                    true_part = grown[part_range]

                    reduced_range = downscales.downscale_slice(part_range, scale)
                    reduced_part = reduced[reduced_range]
                    grown_part = downscales.grow_array(reduced_part, scale, [0], [part_size], [index1%(2**scale)])

                    np.testing.assert_equal(
                        grown_part,
                        true_part,
                        'scale={}, part_range={}'.format(scale, part_range)
                        )

    #########################################
    def test_downscale(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            for data_side in [ 9, 8, 7 ]:
                data = (np.arange(data_side**3) + 1).reshape([data_side]*3).astype(np.uint16)
                for scale in [ 0, 1, 2, 3 ]:
                    reduced = downscales.downscale(data, downsample_kernel, scale)
                    grown = downscales.grow_array(reduced, scale, [0,1,2], data.shape)

                    np.testing.assert_equal(
                            grown.shape, data.shape,
                            'downsample_kernel={}, data_side={}, scale={}'.format(downsample_kernel.name, data_side, scale)
                        )

    #########################################
    def test_predict_new_shape(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            for data_side in [ 9, 8, 7 ]:
                data = (np.arange(data_side**3) + 1).reshape([data_side]*3).astype(np.uint16)
                for scale in [ 0, 1, 2, 3 ]:
                    predicted = downscales.predict_new_shape(data.shape, scale)

                    reduced = downscales.downscale(data, downsample_kernel, scale)
                    np.testing.assert_equal(
                            reduced.shape, predicted,
                            'downsample_kernel={}, data_side={}, scale={}'.format(downsample_kernel.name, data_side, scale)
                        )

                    reduced = data
                    for _ in range(scale):
                        reduced = downscales.downscale(reduced, downsample_kernel, 1)
                    np.testing.assert_equal(
                            reduced.shape, predicted,
                            'downsample_kernel={}, data_side={}, scale={}'.format(downsample_kernel.name, data_side, scale)
                        )

    #########################################
    def test_downscale_in_blocks(self):
        for downsample_kernel in [
                downscales.NullDownsampleKernel(),
                downscales.GaussianDownsampleKernel()
            ]:
            for data_side in [ 17, 16 ]:
                data = (np.arange(data_side**3) + 1).reshape([ data_side ]*3).astype(np.uint16)
                for scale in [ 0, 1, 2 ]:
                    for (batch_size, max_processes) in [
                            (25 if downsample_kernel.name == 'gaussian' else 5, 1),
                            (200, 1),
                            (25 if downsample_kernel.name == 'gaussian' else 5, 2),
                        ]:
                        reduced = downscales.downscale(data, downsample_kernel, scale)
                        reduced_in_blocks = np.zeros(downscales.predict_new_shape(data.shape, scale), data.dtype)

                        downscales.downscale_in_blocks(data, reduced_in_blocks, [ batch_size ]*3, downsample_kernel, scale, max_processes=max_processes)

                        np.testing.assert_equal(
                                reduced,
                                reduced_in_blocks,
                                'downsample_kernel={}, data_side={}, scale={}, batch_size={}, max_processes={}'.format(downsample_kernel.name, data_side, scale, batch_size, max_processes)
                            )

    #########################################
    def test_downscale_pos(self):
        for data_side in [ 9, 8, 7 ]:
            data = (np.arange(data_side**3) + 1).reshape([data_side]*3).astype(np.uint16)
            for scale in [ 0, 1, 2, 3 ]:
                reduced = downscales.downscale(data, downscales.NullDownsampleKernel(), scale)
                grown = downscales.grow_array(reduced, scale, orig_shape=data.shape)

                np.testing.assert_equal(
                        np.array([
                                [
                                        [
                                                reduced[downscales.downscale_pos((slice_index, row, col), scale)]
                                                for col in range(data.shape[2])
                                            ]
                                        for row in range(data.shape[1])
                                    ]
                                for slice_index in range(data.shape[0])
                        ]),
                        grown,
                        'data_side={}'.format(data_side)
                    )

    #########################################
    def test_downscale_slice(self):
        for data_side in [ 9, 8, 7 ]:
            data = np.arange(data_side) + 1
            for scale in [ 0, 1, 2, 3 ]:
                reduced = downscales.downscale(data, downscales.NullDownsampleKernel(), scale)
                grown = downscales.grow_array(reduced, scale, orig_shape=data.shape)
                for start in range(0, data_side-1):
                    for stop in range(start+1, data_side):
                        s = slice(start, stop)
                        np.testing.assert_equal(
                                np.unique(reduced[downscales.downscale_slice(s, scale)]),
                                np.unique(grown[s]),
                                'data_side={}, scale={}, start={}, stop={}'.format(data_side, scale, start, stop)
                            )

if __name__ == '__main__':
    unittest.main()
