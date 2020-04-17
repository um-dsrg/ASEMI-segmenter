import unittest
import numpy as np
import os
import sys
from asemi_segmenter.lib import evaluations

#########################################
class Evaluations(unittest.TestCase):
    
    #########################################
    def test_AccuracyEvaluation(self):
        ev = evaluations.AccuracyEvaluation(2)
        
        np.testing.assert_equal(
            ev.evaluate(
                predicted_labels=np.array([0, 0, 1, 1, 0]),
                true_labels     =np.array([0, 1, 0, 1, 2])
                ),
            ([1/2, 1/2], 2/4)
            )
        np.testing.assert_equal(ev.label_freqs, [2, 2])
        
        np.testing.assert_equal(
            ev.evaluate(
                predicted_labels=np.array([0, 0, 0, 0, 1, 1, 0]),
                true_labels     =np.array([0, 0, 0, 1, 0, 1, 2])
                ),
            ([3/4, 1/2], 4/6)
            )
        np.testing.assert_equal(ev.label_freqs, [6, 4])
        
        np.testing.assert_equal(
            ev.get_global_result_per_label(),
            [4/6, 2/4]
            )
        
        np.testing.assert_equal(
            ev.get_global_result(),
            6/10
            )
    
    #########################################
    def test_IntersectionOverUnionEvaluation(self):
        ev = evaluations.IntersectionOverUnionEvaluation(2)
        
        np.testing.assert_equal(
            ev.evaluate(
                predicted_labels=np.array([0, 0, 1, 1, 0]),
                true_labels     =np.array([0, 1, 0, 1, 2])
                ),
            ([1/3, 1/3], 2/4)
            )
        np.testing.assert_equal(ev.label_freqs, [2, 2])
        
        np.testing.assert_equal(
            ev.evaluate(
                predicted_labels=np.array([0, 0, 0, 0, 1, 1, 0]),
                true_labels     =np.array([0, 0, 0, 1, 0, 1, 2])
                ),
            ([3/5, 1/3], 4/6)
            )
        np.testing.assert_equal(ev.label_freqs, [6, 4])
        
        np.testing.assert_equal(
            ev.get_global_result_per_label(),
            [4/8, 2/6]
            )
        
        np.testing.assert_equal(
            ev.get_global_result(),
            6/10
            )
    
    
if __name__ == '__main__':
    unittest.main()