import unittest
import numpy as np
import os
import sys
from asemi_segmenter.lib import histograms
from asemi_segmenter.lib import downscales
from asemi_segmenter.lib import regions

#########################################
def histogram(array, min_range, max_range, num_bins):
    hist = [ 0 for _ in range(num_bins) ]
    for x in np.nditer(array):
        hist[int((x - min_range)*(num_bins/(max_range - min_range)))] += 1
    return hist

#########################################
class Histograms(unittest.TestCase):
    
    #########################################
    def test_apply_histogram_to_all_neighbourhoods_in_slice_3d(self):
        for data_type in range(2):
            if data_type == 0:
                scaled_data = { 0: (np.arange(4*4*4) + 1).reshape([4,4,4]).astype(np.uint16) }
            else:
                rand = np.random.RandomState(0)
                scaled_data = { 0: rand.randint(0, 2**16, [4,4,4], np.uint16) }
            scaled_data[1] = downscales.downscale(scaled_data[0], downscales.NullDownsampleKernel(), 1)
            
            for scale in [ 1 ]:
                scale_multiple = 2**scale
                for (radius, min_range, max_range, num_bins, row_slice, col_slice) in [
                        (1, 0, scaled_data[scale].max()+1, scaled_data[scale].size+1, slice(None), slice(None)),
                        (5, 0, scaled_data[scale].max()+1, scaled_data[scale].size+1, slice(None), slice(None)),
                    ]:
                    for neighbouring_dims in [ {0,1,2} ]:
                        for slice_index in range(scaled_data[scale].shape[0]):
                            np.testing.assert_equal(
                                    histograms.apply_histogram_to_all_neighbourhoods_in_slice_3d(scaled_data[scale], slice_index, radius, neighbouring_dims, min_range, max_range, num_bins, pad=0, row_slice=row_slice, col_slice=col_slice),
                                    np.array([
                                            [
                                                histogram(regions.get_neighbourhood_array_3d(scaled_data[scale], (slice_index,row,col), radius, neighbouring_dims, pad=0), min_range, max_range, num_bins)
                                                for col in range(col_slice.start or 0, col_slice.stop or scaled_data[scale].shape[2])
                                            ] for row in range(row_slice.start or 0, row_slice.stop or scaled_data[scale].shape[1])
                                        ], np.float32),
                                    'data_type={}, scale={}, radius={}, min_range={}, max_range={}, num_bins={}, row_slice={}, col_slice={}, neighbouring_dims={}, slice_index={}'.format(data_type, scale, radius, min_range, max_range, num_bins, row_slice, col_slice, neighbouring_dims, slice_index)
                                )
    
if __name__ == '__main__':
    unittest.main()
