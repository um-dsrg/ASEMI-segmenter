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
from asemi_segmenter.lib import regions

#########################################
class Regions(unittest.TestCase):

    #########################################
    def test_get_subarray_1d(self):
        data = np.array([1,2,3])

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(1,2)),
                np.array([2])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,3)),
                np.array([1,2,3])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,2)),
                np.array([1,2])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(1,3)),
                np.array([2,3])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(-1,3)),
                np.array([0,1,2,3])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,4)),
                np.array([1,2,3,0])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(-1,4)),
                np.array([0,1,2,3,0])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(-3,-1)),
                np.array([0,0])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(4,5)),
                np.array([0])
            )

        data = np.array([1,2,3,4,5,6,7,8])

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,4), scale=0),
                np.array([1,2,3,4])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,4), scale=1),
                np.array([1,2])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,4), scale=2),
                np.array([1])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(5,8), scale=1),
                np.array([3,4])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(5,8), scale=2),
                np.array([2])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,8), scale=1),
                np.array([1,2,3,4])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,8), scale=2),
                np.array([1,2])
            )

        np.testing.assert_equal(
                regions.get_subarray_1d(data, slice(0,8), scale=3),
                np.array([1])
            )

    #########################################
    def test_get_neighbourhood_array_1d(self):
        data = np.array([1,2,3])

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (0,), 0),
                np.array([1])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (1,), 0),
                np.array([2])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (2,), 0),
                np.array([3])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (-1,), 0),
                np.array([0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 0),
                np.array([0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (0,), 1),
                np.array([0,1,2])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (1,), 1),
                np.array([1,2,3])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (2,), 1),
                np.array([2,3,0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (-1,), 1),
                np.array([0,0,1])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1),
                np.array([3,0,0])
            )

        data = np.array([1,2,3,4,5,6,7,8])

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1, scale=0, scale_radius=False),
                np.array([3,4,5])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1, scale=0, scale_radius=True),
                np.array([3,4,5])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1, scale=1, scale_radius=False),
                np.array([1,2,3])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1, scale=1, scale_radius=True),
                np.array([2])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1, scale=2, scale_radius=False),
                np.array([0,1,2])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 1, scale=2, scale_radius=True),
                np.array([1])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (7,), 1, scale=0, scale_radius=False),
                np.array([7,8,0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (7,), 1, scale=0, scale_radius=True),
                np.array([7,8,0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (7,), 1, scale=1, scale_radius=False),
                np.array([3,4,5])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (7,), 1, scale=1, scale_radius=True),
                np.array([4])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (7,), 1, scale=2, scale_radius=False),
                np.array([1,2,3])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (7,), 1, scale=2, scale_radius=True),
                np.array([2])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 2, scale=0, scale_radius=False),
                np.array([2,3,4,5,6])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 2, scale=0, scale_radius=True),
                np.array([2,3,4,5,6])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 2, scale=1, scale_radius=False),
                np.array([0,1,2,3,4])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 2, scale=1, scale_radius=True),
                np.array([1,2,3])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 2, scale=2, scale_radius=False),
                np.array([0,0,1,2,3])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_1d(data, (3,), 2, scale=2, scale_radius=True),
                np.array([1])
            )

    #########################################
    def test_get_subarray_2d(self):
        data = np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
            ])

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(1,2), slice(1,2)),
                np.array([
                        [5],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,3), slice(0,3)),
                np.array([
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,2), slice(0,3)),
                np.array([
                        [1,2,3],
                        [4,5,6],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,3), slice(0,2)),
                np.array([
                        [1,2],
                        [4,5],
                        [7,8],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,2), slice(0,2)),
                np.array([
                        [1,2],
                        [4,5],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(1,3), slice(0,3)),
                np.array([
                        [4,5,6],
                        [7,8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,3), slice(1,3)),
                np.array([
                        [2,3],
                        [5,6],
                        [8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(1,3), slice(1,3)),
                np.array([
                        [5,6],
                        [8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(1,2), slice(0,3)),
                np.array([
                        [4,5,6],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,3), slice(1,2)),
                np.array([
                        [2],
                        [5],
                        [8],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(-1,3), slice(0,3)),
                np.array([
                        [0,0,0],
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,4), slice(0,3)),
                np.array([
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                        [0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,3), slice(-1,3)),
                np.array([
                        [0,1,2,3],
                        [0,4,5,6],
                        [0,7,8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,3), slice(0,4)),
                np.array([
                        [1,2,3,0],
                        [4,5,6,0],
                        [7,8,9,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(-1,4), slice(-1,4)),
                np.array([
                        [0,0,0,0,0],
                        [0,1,2,3,0],
                        [0,4,5,6,0],
                        [0,7,8,9,0],
                        [0,0,0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(-3,-1), slice(-3,-1)),
                np.array([
                        [0,0],
                        [0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(4,5), slice(4,5)),
                np.array([
                        [0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(-3,-1), slice(4,5)),
                np.array([
                        [0],
                        [0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(-3,-2), slice(0,2)),
                np.array([
                        [0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(1,2), slice(-3,-1)),
                np.array([
                        [0,0],
                    ])
            )

        data = np.array([
                [ 1, 2, 3, 4, 5, 6, 7, 8],
                [ 9,10,11,12,13,14,15,16],
                [17,18,19,20,21,22,23,24],
                [25,26,27,28,29,30,31,32],
                [33,34,35,36,37,38,39,40],
                [41,42,43,44,45,46,47,48],
                [49,50,51,52,53,54,55,56],
                [57,58,59,60,61,62,63,64],
            ])

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,4), slice(0,4), scale=0),
                np.array([
                    [ 1, 2, 3, 4],
                    [ 9,10,11,12],
                    [17,18,19,20],
                    [25,26,27,28],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,4), slice(0,4), scale=1),
                np.array([
                    [ 1, 2],
                    [ 9,10],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,4), slice(0,4), scale=2),
                np.array([
                    [ 1],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(5,8), slice(5,8), scale=1),
                np.array([
                    [19,20],
                    [27,28],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(5,8), slice(5,8), scale=2),
                np.array([
                    [10],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,8), slice(0,8), scale=1),
                np.array([
                    [ 1, 2, 3, 4],
                    [ 9,10,11,12],
                    [17,18,19,20],
                    [25,26,27,28],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,8), slice(0,8), scale=2),
                np.array([
                    [ 1, 2],
                    [ 9,10],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_2d(data, slice(0,8), slice(0,8), scale=3),
                np.array([
                    [ 1],
                ])
            )

    #########################################
    def test_get_neighbourhood_array_2d(self):
        data = np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
            ])

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (0,0), 0, {0,1}),
                np.array([
                        [1],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (1,1), 0, {0,1}),
                np.array([
                        [5],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,2), 0, {0,1}),
                np.array([
                        [9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,2), 0, {0,1}),
                np.array([
                        [0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,3), 0, {0,1}),
                np.array([
                        [0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 0, {0,1}),
                np.array([
                        [0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (0,0), 1, {0,1}),
                np.array([
                        [0,0,0],
                        [0,1,2],
                        [0,4,5],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (1,1), 1, {0,1}),
                np.array([
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,2), 1, {0,1}),
                np.array([
                        [5,6,0],
                        [8,9,0],
                        [0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,3), 1, {0,1}),
                np.array([
                        [6,0,0],
                        [9,0,0],
                        [0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,2), 1, {0,1}),
                np.array([
                        [8,9,0],
                        [0,0,0],
                        [0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}),
                np.array([
                        [9,0,0],
                        [0,0,0],
                        [0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (0,0), 2, {0,1}),
                np.array([
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,1,2,3],
                        [0,0,4,5,6],
                        [0,0,7,8,9],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (1,1), 2, {0,1}),
                np.array([
                        [0,0,0,0,0],
                        [0,1,2,3,0],
                        [0,4,5,6,0],
                        [0,7,8,9,0],
                        [0,0,0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,2), 2, {0,1}),
                np.array([
                        [1,2,3,0,0],
                        [4,5,6,0,0],
                        [7,8,9,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0],
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,2), 2, {0}),
                np.array([3,6,9,0,0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (2,2), 2, {1}),
                np.array([7,8,9,0,0])
            )

        data = np.array([
                [ 1, 2, 3, 4, 5, 6, 7, 8],
                [ 9,10,11,12,13,14,15,16],
                [17,18,19,20,21,22,23,24],
                [25,26,27,28,29,30,31,32],
                [33,34,35,36,37,38,39,40],
                [41,42,43,44,45,46,47,48],
                [49,50,51,52,53,54,55,56],
                [57,58,59,60,61,62,63,64],
            ])

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}, scale=0, scale_radius=False),
                np.array([
                    [19,20,21],
                    [27,28,29],
                    [35,36,37],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}, scale=0, scale_radius=True),
                np.array([
                    [19,20,21],
                    [27,28,29],
                    [35,36,37],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}, scale=1, scale_radius=False),
                np.array([
                    [ 1, 2, 3],
                    [ 9,10,11],
                    [17,18,19],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}, scale=1, scale_radius=True),
                np.array([
                    [10],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}, scale=2, scale_radius=False),
                np.array([
                    [ 0, 0, 0],
                    [ 0, 1, 2],
                    [ 0, 9,10],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 1, {0,1}, scale=2, scale_radius=True),
                np.array([
                    [ 1],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (7,7), 1, {0,1}, scale=0, scale_radius=False),
                np.array([
                    [55,56, 0],
                    [63,64, 0],
                    [ 0, 0, 0],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (7,7), 1, {0,1}, scale=0, scale_radius=True),
                np.array([
                    [55,56, 0],
                    [63,64, 0],
                    [ 0, 0, 0],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (7,7), 1, {0,1}, scale=1, scale_radius=False),
                np.array([
                    [19,20,21],
                    [27,28,29],
                    [35,36,37],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (7,7), 1, {0,1}, scale=1, scale_radius=True),
                np.array([
                    [28],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (7,7), 1, {0,1}, scale=2, scale_radius=False),
                np.array([
                    [ 1, 2, 3],
                    [ 9,10,11],
                    [17,18,19],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (7,7), 1, {0,1}, scale=2, scale_radius=True),
                np.array([
                    [10],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 2, {0,1}, scale=0, scale_radius=False),
                np.array([
                    [10,11,12,13,14],
                    [18,19,20,21,22],
                    [26,27,28,29,30],
                    [34,35,36,37,38],
                    [42,43,44,45,46],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 2, {0,1}, scale=0, scale_radius=True),
                np.array([
                    [10,11,12,13,14],
                    [18,19,20,21,22],
                    [26,27,28,29,30],
                    [34,35,36,37,38],
                    [42,43,44,45,46],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 2, {0,1}, scale=1, scale_radius=False),
                np.array([
                    [ 0, 0, 0, 0, 0],
                    [ 0, 1, 2, 3, 4],
                    [ 0, 9,10,11,12],
                    [ 0,17,18,19,20],
                    [ 0,25,26,27,28],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 2, {0,1}, scale=1, scale_radius=True),
                np.array([
                    [ 1, 2, 3],
                    [ 9,10,11],
                    [17,18,19],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 2, {0,1}, scale=2, scale_radius=False),
                np.array([
                    [ 0, 0, 0, 0, 0],
                    [ 0, 0, 0, 0, 0],
                    [ 0, 0, 1, 2, 3],
                    [ 0, 0, 9,10,11],
                    [ 0, 0,17,18,19],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_2d(data, (3,3), 2, {0,1}, scale=2, scale_radius=True),
                np.array([
                    [ 1],
                ])
            )

    #########################################
    def test_get_subarray_3d(self):
        data = np.array([
                [
                    [ 1, 2, 3],
                    [ 4, 5, 6],
                    [ 7, 8, 9],
                ],
                [
                    [10,11,12],
                    [13,14,15],
                    [16,17,18],
                ],
                [
                    [19,20,21],
                    [22,23,24],
                    [25,26,27],
                ],
            ])

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(1,2), slice(1,2), slice(1,2)),
                np.array([
                        [
                            [14],
                        ]
                    ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,2), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,2), slice(0,3)),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(0,2)),
                np.array([
                    [
                        [ 1, 2],
                        [ 4, 5],
                        [ 7, 8],
                    ],
                    [
                        [10,11],
                        [13,14],
                        [16,17],
                    ],
                    [
                        [19,20],
                        [22,23],
                        [25,26],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(1,3), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(1,3), slice(0,3)),
                np.array([
                    [
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [22,23,24],
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(1,3)),
                np.array([
                    [
                        [ 2, 3],
                        [ 5, 6],
                        [ 8, 9],
                    ],
                    [
                        [11,12],
                        [14,15],
                        [17,18],
                    ],
                    [
                        [20,21],
                        [23,24],
                        [26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,2), slice(0,3), slice(0,2)),
                np.array([
                    [
                        [ 1, 2],
                        [ 4, 5],
                        [ 7, 8],
                    ],
                    [
                        [10,11],
                        [13,14],
                        [16,17],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(1,3), slice(0,2)),
                np.array([
                    [
                        [ 4, 5],
                        [ 7, 8],
                    ],
                    [
                        [13,14],
                        [16,17],
                    ],
                    [
                        [22,23],
                        [25,26],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,1), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(2,3), slice(0,3)),
                np.array([
                    [
                        [ 7, 8, 9],
                    ],
                    [
                        [16,17,18],
                    ],
                    [
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(1,2)),
                np.array([
                    [
                        [ 2],
                        [ 5],
                        [ 8],
                    ],
                    [
                        [11],
                        [14],
                        [17],
                    ],
                    [
                        [20],
                        [23],
                        [26],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,1), slice(0,3), slice(0,3), to_2d=True),
                np.array([
                    [ 1, 2, 3],
                    [ 4, 5, 6],
                    [ 7, 8, 9],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(2,3), slice(0,3), to_2d=True),
                np.array([
                    [ 7, 8, 9],
                    [16,17,18],
                    [25,26,27],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(1,2), to_2d=True),
                np.array([
                    [ 2, 5, 8],
                    [11,14,17],
                    [20,23,26],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,1), slice(0,2), slice(1,3), to_2d=True),
                np.array([
                    [ 2, 3],
                    [ 5, 6],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(1,3), slice(2,3), slice(1,3), to_2d=True),
                np.array([
                    [17,18],
                    [26,27],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,2), slice(0,2), slice(1,2), to_2d=True),
                np.array([
                    [ 2, 5],
                    [11,14],
                ])
            )

        with self.assertRaises(ValueError):
            regions.get_subarray_3d(data, slice(0,1), slice(0,1), slice(0,3), to_2d=True)
        with self.assertRaises(ValueError):
            regions.get_subarray_3d(data, slice(0,3), slice(0,1), slice(2,3), to_2d=True)
        with self.assertRaises(ValueError):
            regions.get_subarray_3d(data, slice(2,3), slice(0,3), slice(0,1), to_2d=True)

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(-1,3), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(-1,3), slice(0,3)),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [ 0, 0, 0],
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [ 0, 0, 0],
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(-1,3)),
                np.array([
                    [
                        [ 0, 1, 2, 3],
                        [ 0, 4, 5, 6],
                        [ 0, 7, 8, 9],
                    ],
                    [
                        [ 0,10,11,12],
                        [ 0,13,14,15],
                        [ 0,16,17,18],
                    ],
                    [
                        [ 0,19,20,21],
                        [ 0,22,23,24],
                        [ 0,25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,4), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,4), slice(0,3)),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                        [ 0, 0, 0],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                        [ 0, 0, 0],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(0,4)),
                np.array([
                    [
                        [ 1, 2, 3, 0],
                        [ 4, 5, 6, 0],
                        [ 7, 8, 9, 0],
                    ],
                    [
                        [10,11,12, 0],
                        [13,14,15, 0],
                        [16,17,18, 0],
                    ],
                    [
                        [19,20,21, 0],
                        [22,23,24, 0],
                        [25,26,27, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(-1,4), slice(-1,4), slice(-1,4)),
                np.array([
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0, 1, 2, 3, 0],
                        [ 0, 4, 5, 6, 0],
                        [ 0, 7, 8, 9, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0,10,11,12, 0],
                        [ 0,13,14,15, 0],
                        [ 0,16,17,18, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0,19,20,21, 0],
                        [ 0,22,23,24, 0],
                        [ 0,25,26,27, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(-3,-1), slice(0,3), slice(0,3)),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(3,5), slice(0,3)),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(0,3), slice(-5,-3)),
                np.array([
                    [
                        [ 0, 0],
                        [ 0, 0],
                        [ 0, 0],
                    ],
                    [
                        [ 0, 0],
                        [ 0, 0],
                        [ 0, 0],
                    ],
                    [
                        [ 0, 0],
                        [ 0, 0],
                        [ 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(3,4), slice(3,4), slice(3,4)),
                np.array([
                    [
                        [ 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,1), slice(0,4), slice(0,3), to_2d=True),
                np.array([
                    [ 1, 2, 3],
                    [ 4, 5, 6],
                    [ 7, 8, 9],
                    [ 0, 0, 0],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,3), slice(1,2), slice(-1,4), to_2d=True),
                np.array([
                    [ 0, 4, 5, 6, 0],
                    [ 0,13,14,15, 0],
                    [ 0,22,23,24, 0],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(-1,3), slice(0,3), slice(2,3), to_2d=True),
                np.array([
                    [ 0, 0, 0],
                    [ 3, 6, 9],
                    [12,15,18],
                    [21,24,27],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(-3,-1), slice(0,1), slice(0,3), to_2d=True),
                np.array([
                    [ 0, 0, 0],
                    [ 0, 0, 0],
                ])
            )

        with self.assertRaises(ValueError):
            regions.get_subarray_3d(data, slice(0,1), slice(0,1), slice(-3,-1), to_2d=True)
        with self.assertRaises(ValueError):
            regions.get_subarray_3d(data, slice(3,5), slice(0,1), slice(2,3), to_2d=True)
        with self.assertRaises(ValueError):
            regions.get_subarray_3d(data, slice(2,3), slice(7,9), slice(0,1), to_2d=True)

        data = np.array([
                [
                    [  1,  2,  3,  4,  5,  6,  7,  8,],
                    [  9, 10, 11, 12, 13, 14, 15, 16,],
                    [ 17, 18, 19, 20, 21, 22, 23, 24,],
                    [ 25, 26, 27, 28, 29, 30, 31, 32,],
                    [ 33, 34, 35, 36, 37, 38, 39, 40,],
                    [ 41, 42, 43, 44, 45, 46, 47, 48,],
                    [ 49, 50, 51, 52, 53, 54, 55, 56,],
                    [ 57, 58, 59, 60, 61, 62, 63, 64,],
                ],
                [
                    [ 65, 66, 67, 68, 69, 70, 71, 72,],
                    [ 73, 74, 75, 76, 77, 78, 79, 80,],
                    [ 81, 82, 83, 84, 85, 86, 87, 88,],
                    [ 89, 90, 91, 92, 93, 94, 95, 96,],
                    [ 97, 98, 99,100,101,102,103,104,],
                    [105,106,107,108,109,110,111,112,],
                    [113,114,115,116,117,118,119,120,],
                    [121,122,123,124,125,126,127,128,],
                ],
                [
                    [129,130,131,132,133,134,135,136,],
                    [137,138,139,140,141,142,143,144,],
                    [145,146,147,148,149,150,151,152,],
                    [153,154,155,156,157,158,159,160,],
                    [161,162,163,164,165,166,167,168,],
                    [169,170,171,172,173,174,175,176,],
                    [177,178,179,180,181,182,183,184,],
                    [185,186,187,188,189,190,191,192,],
                ],
                [
                    [193,194,195,196,197,198,199,200,],
                    [201,202,203,204,205,206,207,208,],
                    [209,210,211,212,213,214,215,216,],
                    [217,218,219,220,221,222,223,224,],
                    [225,226,227,228,229,230,231,232,],
                    [233,234,235,236,237,238,239,240,],
                    [241,242,243,244,245,246,247,248,],
                    [249,250,251,252,253,254,255,256,],
                ],
                [
                    [257,258,259,260,261,262,263,264,],
                    [265,266,267,268,269,270,271,272,],
                    [273,274,275,276,277,278,279,280,],
                    [281,282,283,284,285,286,287,288,],
                    [289,290,291,292,293,294,295,296,],
                    [297,298,299,300,301,302,303,304,],
                    [305,306,307,308,309,310,311,312,],
                    [313,314,315,316,317,318,319,320,],
                ],
                [
                    [321,322,323,324,325,326,327,328,],
                    [329,330,331,332,333,334,335,336,],
                    [337,338,339,340,341,342,343,344,],
                    [345,346,347,348,349,350,351,352,],
                    [353,354,355,356,357,358,359,360,],
                    [361,362,363,364,365,366,367,368,],
                    [369,370,371,372,373,374,375,376,],
                    [377,378,379,380,381,382,383,384,],
                ],
                [
                    [385,386,387,388,389,390,391,392,],
                    [393,394,395,396,397,398,399,400,],
                    [401,402,403,404,405,406,407,408,],
                    [409,410,411,412,413,414,415,416,],
                    [417,418,419,420,421,422,423,424,],
                    [425,426,427,428,429,430,431,432,],
                    [433,434,435,436,437,438,439,440,],
                    [441,442,443,444,445,446,447,448,],
                ],
                [
                    [449,450,451,452,453,454,455,456,],
                    [457,458,459,460,461,462,463,464,],
                    [465,466,467,468,469,470,471,472,],
                    [473,474,475,476,477,478,479,480,],
                    [481,482,483,484,485,486,487,488,],
                    [489,490,491,492,493,494,495,496,],
                    [497,498,499,500,501,502,503,504,],
                    [505,506,507,508,509,510,511,512,],
                ]
            ])

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,4), slice(0,4), slice(0,4), scale=0),
                np.array([
                    [
                        [  1,  2,  3,  4],
                        [  9, 10, 11, 12],
                        [ 17, 18, 19, 20],
                        [ 25, 26, 27, 28],
                    ],
                    [
                        [ 65, 66, 67, 68],
                        [ 73, 74, 75, 76],
                        [ 81, 82, 83, 84],
                        [ 89, 90, 91, 92],
                    ],
                    [
                        [129,130,131,132],
                        [137,138,139,140],
                        [145,146,147,148],
                        [153,154,155,156],
                    ],
                    [
                        [193,194,195,196],
                        [201,202,203,204],
                        [209,210,211,212],
                        [217,218,219,220],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,4), slice(0,4), slice(0,4), scale=1),
                np.array([
                    [
                        [  1,  2],
                        [  9, 10],
                    ],
                    [
                        [ 65, 66],
                        [ 73, 74],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,4), slice(0,4), slice(0,4), scale=2),
                np.array([
                    [
                        [  1],
                    ]
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(5,8), slice(5,8), slice(5,8), scale=1),
                np.array([
                    [
                        [147,148],
                        [155,156],
                    ],
                    [
                        [211,212],
                        [219,220],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(5,8), slice(5,8), slice(5,8), scale=2),
                np.array([
                    [
                        [ 74],
                    ]
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,8), slice(0,8), slice(0,8), scale=1),
                np.array([
                    [
                        [  1,  2,  3,  4],
                        [  9, 10, 11, 12],
                        [ 17, 18, 19, 20],
                        [ 25, 26, 27, 28],
                    ],
                    [
                        [ 65, 66, 67, 68],
                        [ 73, 74, 75, 76],
                        [ 81, 82, 83, 84],
                        [ 89, 90, 91, 92],
                    ],
                    [
                        [129,130,131,132],
                        [137,138,139,140],
                        [145,146,147,148],
                        [153,154,155,156],
                    ],
                    [
                        [193,194,195,196],
                        [201,202,203,204],
                        [209,210,211,212],
                        [217,218,219,220],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,8), slice(0,8), slice(0,8), scale=2),
                np.array([
                    [
                        [  1,  2],
                        [  9, 10],
                    ],
                    [
                        [ 65, 66],
                        [ 73, 74],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_subarray_3d(data, slice(0,8), slice(0,8), slice(0,8), scale=3),
                np.array([
                    [
                        [  1],
                    ]
                ])
            )

    #########################################
    def test_get_neighbourhood_array_3d(self):
        data = np.array([
                [
                    [ 1, 2, 3],
                    [ 4, 5, 6],
                    [ 7, 8, 9],
                ],
                [
                    [10,11,12],
                    [13,14,15],
                    [16,17,18],
                ],
                [
                    [19,20,21],
                    [22,23,24],
                    [25,26,27],
                ],
            ])

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 0, {0,1,2}),
                np.array([
                        [
                            [ 1],
                        ]
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 0, {0,1,2}),
                np.array([
                        [
                            [14],
                        ]
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,2), 0, {0,1,2}),
                np.array([
                        [
                            [27],
                        ]
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 0, {0,1,2}),
                np.array([
                        [
                            [ 0],
                        ]
                    ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 1, {0,1,2}),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 1, 2],
                        [ 0, 4, 5],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0,10,11],
                        [ 0,13,14],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {0,1,2}),
                np.array([
                    [
                        [ 1, 2, 3],
                        [ 4, 5, 6],
                        [ 7, 8, 9],
                    ],
                    [
                        [10,11,12],
                        [13,14,15],
                        [16,17,18],
                    ],
                    [
                        [19,20,21],
                        [22,23,24],
                        [25,26,27],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,2), 1, {0,1,2}),
                np.array([
                    [
                        [14,15, 0],
                        [17,18, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [23,24, 0],
                        [26,27, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,3), 1, {0,1,2}),
                np.array([
                    [
                        [15, 0, 0],
                        [18, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [24, 0, 0],
                        [27, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,4), 1, {0,1,2}),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (4,4,4), 1, {0,1,2}),
                np.array([
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                        [ 0, 0, 0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {0,1}),
                np.array([
                    [ 2, 5, 8],
                    [11,14,17],
                    [20,23,26],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {0,2}),
                np.array([
                    [ 4, 5, 6],
                    [13,14,15],
                    [22,23,24],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {1,2}),
                np.array([
                    [10,11,12],
                    [13,14,15],
                    [16,17,18],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 1, {0,1}),
                np.array([
                    [ 0, 0, 0],
                    [ 0, 1, 4],
                    [ 0,10,13],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 1, {0,2}),
                np.array([
                    [ 0, 0, 0],
                    [ 0, 1, 2],
                    [ 0,10,11],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,2), 1, {1,2}),
                np.array([
                    [23,24, 0],
                    [26,27, 0],
                    [ 0, 0, 0],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 1, {0}),
                np.array([ 0, 1,10])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {0}),
                np.array([ 5,14,23])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,2), 1, {0}),
                np.array([18,27, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,3), 1, {0}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,4), 1, {0}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (4,4,4), 1, {0}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 1, {1}),
                np.array([ 0, 1, 4])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {1}),
                np.array([11,14,17])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,2), 1, {1}),
                np.array([24,27, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,3), 1, {1}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,4), 1, {1}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (4,4,4), 1, {1}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (0,0,0), 1, {2}),
                np.array([ 0, 1, 2])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 1, {2}),
                np.array([13,14,15])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,2), 1, {2}),
                np.array([26,27, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,3), 1, {2}),
                np.array([27, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (2,2,4), 1, {2}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (4,4,4), 1, {2}),
                np.array([ 0, 0, 0])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (1,1,1), 2, {0,1,2}),
                np.array([
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0, 1, 2, 3, 0],
                        [ 0, 4, 5, 6, 0],
                        [ 0, 7, 8, 9, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0,10,11,12, 0],
                        [ 0,13,14,15, 0],
                        [ 0,16,17,18, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0,19,20,21, 0],
                        [ 0,22,23,24, 0],
                        [ 0,25,26,27, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                    [
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0],
                    ],
                ])
            )

        data = np.array([
                [
                    [  1,  2,  3,  4,  5,  6,  7,  8,],
                    [  9, 10, 11, 12, 13, 14, 15, 16,],
                    [ 17, 18, 19, 20, 21, 22, 23, 24,],
                    [ 25, 26, 27, 28, 29, 30, 31, 32,],
                    [ 33, 34, 35, 36, 37, 38, 39, 40,],
                    [ 41, 42, 43, 44, 45, 46, 47, 48,],
                    [ 49, 50, 51, 52, 53, 54, 55, 56,],
                    [ 57, 58, 59, 60, 61, 62, 63, 64,],
                ],
                [
                    [ 65, 66, 67, 68, 69, 70, 71, 72,],
                    [ 73, 74, 75, 76, 77, 78, 79, 80,],
                    [ 81, 82, 83, 84, 85, 86, 87, 88,],
                    [ 89, 90, 91, 92, 93, 94, 95, 96,],
                    [ 97, 98, 99,100,101,102,103,104,],
                    [105,106,107,108,109,110,111,112,],
                    [113,114,115,116,117,118,119,120,],
                    [121,122,123,124,125,126,127,128,],
                ],
                [
                    [129,130,131,132,133,134,135,136,],
                    [137,138,139,140,141,142,143,144,],
                    [145,146,147,148,149,150,151,152,],
                    [153,154,155,156,157,158,159,160,],
                    [161,162,163,164,165,166,167,168,],
                    [169,170,171,172,173,174,175,176,],
                    [177,178,179,180,181,182,183,184,],
                    [185,186,187,188,189,190,191,192,],
                ],
                [
                    [193,194,195,196,197,198,199,200,],
                    [201,202,203,204,205,206,207,208,],
                    [209,210,211,212,213,214,215,216,],
                    [217,218,219,220,221,222,223,224,],
                    [225,226,227,228,229,230,231,232,],
                    [233,234,235,236,237,238,239,240,],
                    [241,242,243,244,245,246,247,248,],
                    [249,250,251,252,253,254,255,256,],
                ],
                [
                    [257,258,259,260,261,262,263,264,],
                    [265,266,267,268,269,270,271,272,],
                    [273,274,275,276,277,278,279,280,],
                    [281,282,283,284,285,286,287,288,],
                    [289,290,291,292,293,294,295,296,],
                    [297,298,299,300,301,302,303,304,],
                    [305,306,307,308,309,310,311,312,],
                    [313,314,315,316,317,318,319,320,],
                ],
                [
                    [321,322,323,324,325,326,327,328,],
                    [329,330,331,332,333,334,335,336,],
                    [337,338,339,340,341,342,343,344,],
                    [345,346,347,348,349,350,351,352,],
                    [353,354,355,356,357,358,359,360,],
                    [361,362,363,364,365,366,367,368,],
                    [369,370,371,372,373,374,375,376,],
                    [377,378,379,380,381,382,383,384,],
                ],
                [
                    [385,386,387,388,389,390,391,392,],
                    [393,394,395,396,397,398,399,400,],
                    [401,402,403,404,405,406,407,408,],
                    [409,410,411,412,413,414,415,416,],
                    [417,418,419,420,421,422,423,424,],
                    [425,426,427,428,429,430,431,432,],
                    [433,434,435,436,437,438,439,440,],
                    [441,442,443,444,445,446,447,448,],
                ],
                [
                    [449,450,451,452,453,454,455,456,],
                    [457,458,459,460,461,462,463,464,],
                    [465,466,467,468,469,470,471,472,],
                    [473,474,475,476,477,478,479,480,],
                    [481,482,483,484,485,486,487,488,],
                    [489,490,491,492,493,494,495,496,],
                    [497,498,499,500,501,502,503,504,],
                    [505,506,507,508,509,510,511,512,],
                ]
            ])

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 1, {0,1,2}, scale=0, scale_radius=False),
                np.array([
                    [
                        [147,148,149],
                        [155,156,157],
                        [163,164,165],
                    ],
                    [
                        [211,212,213],
                        [219,220,221],
                        [227,228,229],
                    ],
                    [
                        [275,276,277],
                        [283,284,285],
                        [291,292,293],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 1, {0,1,2}, scale=0, scale_radius=True),
                np.array([
                    [
                        [147,148,149],
                        [155,156,157],
                        [163,164,165],
                    ],
                    [
                        [211,212,213],
                        [219,220,221],
                        [227,228,229],
                    ],
                    [
                        [275,276,277],
                        [283,284,285],
                        [291,292,293],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 1, {0,1,2}, scale=1, scale_radius=False),
                np.array([
                    [
                        [  1,  2,  3],
                        [  9, 10, 11],
                        [ 17, 18, 19],
                    ],
                    [
                        [ 65, 66, 67],
                        [ 73, 74, 75],
                        [ 81, 82, 83],
                    ],
                    [
                        [129,130,131],
                        [137,138,139],
                        [145,146,147],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 1, {0,1,2}, scale=1, scale_radius=True),
                np.array([
                    [
                        [74],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 1, {0,1,2}, scale=2, scale_radius=False),
                np.array([
                    [
                        [  0,  0,  0],
                        [  0,  0,  0],
                        [  0,  0,  0],
                    ],
                    [
                        [  0,  0,  0],
                        [  0,  1,  2],
                        [  0,  9, 10],
                    ],
                    [
                        [  0,  0,  0],
                        [  0, 65, 66],
                        [  0, 73, 74],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 1, {0,1,2}, scale=2, scale_radius=True),
                np.array([
                    [
                        [ 1],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (7,7,7), 1, {0,1,2}, scale=0, scale_radius=False),
                np.array([
                    [
                        [439,440,  0],
                        [447,448,  0],
                        [  0,  0,  0],
                    ],
                    [
                        [503,504,  0],
                        [511,512,  0],
                        [  0,  0,  0],
                    ],
                    [
                        [  0,  0,  0],
                        [  0,  0,  0],
                        [  0,  0,  0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (7,7,7), 1, {0,1,2}, scale=0, scale_radius=True),
                np.array([
                    [
                        [439,440,  0],
                        [447,448,  0],
                        [  0,  0,  0],
                    ],
                    [
                        [503,504,  0],
                        [511,512,  0],
                        [  0,  0,  0],
                    ],
                    [
                        [  0,  0,  0],
                        [  0,  0,  0],
                        [  0,  0,  0],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (7,7,7), 1, {0,1,2}, scale=1, scale_radius=False),
                np.array([
                    [
                        [147,148,149],
                        [155,156,157],
                        [163,164,165],
                    ],
                    [
                        [211,212,213],
                        [219,220,221],
                        [227,228,229],
                    ],
                    [
                        [275,276,277],
                        [283,284,285],
                        [291,292,293],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (7,7,7), 1, {0,1,2}, scale=1, scale_radius=True),
                np.array([
                    [
                        [220],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (7,7,7), 1, {0,1,2}, scale=2, scale_radius=False),
                np.array([
                    [
                        [  1,  2,  3],
                        [  9, 10, 11],
                        [ 17, 18, 19],
                    ],
                    [
                        [ 65, 66, 67],
                        [ 73, 74, 75],
                        [ 81, 82, 83],
                    ],
                    [
                        [129,130,131],
                        [137,138,139],
                        [145,146,147],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (7,7,7), 1, {0,1,2}, scale=2, scale_radius=True),
                np.array([
                    [
                        [ 74],
                    ]
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 2, {0,1,2}, scale=0, scale_radius=False),
                np.array([
                    [
                        [ 74, 75, 76, 77, 78],
                        [ 82, 83, 84, 85, 86],
                        [ 90, 91, 92, 93, 94],
                        [ 98, 99,100,101,102],
                        [106,107,108,109,110],
                    ],
                    [
                        [138,139,140,141,142],
                        [146,147,148,149,150],
                        [154,155,156,157,158],
                        [162,163,164,165,166],
                        [170,171,172,173,174],
                    ],
                    [
                        [202,203,204,205,206],
                        [210,211,212,213,214],
                        [218,219,220,221,222],
                        [226,227,228,229,230],
                        [234,235,236,237,238],
                    ],
                    [
                        [266,267,268,269,270],
                        [274,275,276,277,278],
                        [282,283,284,285,286],
                        [290,291,292,293,294],
                        [298,299,300,301,302],
                    ],
                    [
                        [330,331,332,333,334],
                        [338,339,340,341,342],
                        [346,347,348,349,350],
                        [354,355,356,357,358],
                        [362,363,364,365,366],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 2, {0,1,2}, scale=0, scale_radius=True),
                np.array([
                    [
                        [ 74, 75, 76, 77, 78],
                        [ 82, 83, 84, 85, 86],
                        [ 90, 91, 92, 93, 94],
                        [ 98, 99,100,101,102],
                        [106,107,108,109,110],
                    ],
                    [
                        [138,139,140,141,142],
                        [146,147,148,149,150],
                        [154,155,156,157,158],
                        [162,163,164,165,166],
                        [170,171,172,173,174],
                    ],
                    [
                        [202,203,204,205,206],
                        [210,211,212,213,214],
                        [218,219,220,221,222],
                        [226,227,228,229,230],
                        [234,235,236,237,238],
                    ],
                    [
                        [266,267,268,269,270],
                        [274,275,276,277,278],
                        [282,283,284,285,286],
                        [290,291,292,293,294],
                        [298,299,300,301,302],
                    ],
                    [
                        [330,331,332,333,334],
                        [338,339,340,341,342],
                        [346,347,348,349,350],
                        [354,355,356,357,358],
                        [362,363,364,365,366],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 2, {0,1,2}, scale=1, scale_radius=False),
                np.array([
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  1,  2,  3,  4],
                        [  0,  9, 10, 11, 12],
                        [  0, 17, 18, 19, 20],
                        [  0, 25, 26, 27, 28],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0, 65, 66, 67, 68],
                        [  0, 73, 74, 75, 76],
                        [  0, 81, 82, 83, 84],
                        [  0, 89, 90, 91, 92],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,129,130,131,132],
                        [  0,137,138,139,140],
                        [  0,145,146,147,148],
                        [  0,153,154,155,156],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,193,194,195,196],
                        [  0,201,202,203,204],
                        [  0,209,210,211,212],
                        [  0,217,218,219,220],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 2, {0,1,2}, scale=1, scale_radius=True),
                np.array([
                    [
                        [  1,  2,  3],
                        [  9, 10, 11],
                        [ 17, 18, 19],
                    ],
                    [
                        [ 65, 66, 67],
                        [ 73, 74, 75],
                        [ 81, 82, 83],
                    ],
                    [
                        [129,130,131],
                        [137,138,139],
                        [145,146,147],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 2, {0,1,2}, scale=2, scale_radius=False),
                np.array([
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  1,  2,  3],
                        [  0,  0,  9, 10, 11],
                        [  0,  0, 17, 18, 19],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0, 65, 66, 67],
                        [  0,  0, 73, 74, 75],
                        [  0,  0, 81, 82, 83],
                    ],
                    [
                        [  0,  0,  0,  0,  0],
                        [  0,  0,  0,  0,  0],
                        [  0,  0,129,130,131],
                        [  0,  0,137,138,139],
                        [  0,  0,145,146,147],
                    ],
                ])
            )

        np.testing.assert_equal(
                regions.get_neighbourhood_array_3d(data, (3,3,3), 2, {0,1,2}, scale=2, scale_radius=True),
                np.array([
                    [
                        [  1],
                    ],
                ])
            )

if __name__ == '__main__':
    unittest.main()
