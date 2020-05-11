import unittest
import numpy as np
import os
import sys
from asemi_segmenter.lib import evaluations
from asemi_segmenter.lib import volumes

#########################################
class Evaluations(unittest.TestCase):
    
    #########################################
    def test_get_confusion_matrix(self):
        np.testing.assert_equal(
            evaluations.get_confusion_matrix(
                predicted_labels=np.array([0, 0, 1, 1]),
                true_labels     =np.array([0, 1, 0, 1]),
                num_labels=2
                ),
            np.array([
                [1, 1],
                [1, 1]
            ])
            )
        
        np.testing.assert_equal(
            evaluations.get_confusion_matrix(
                predicted_labels=np.array([0, 0, 1, 1]),
                true_labels     =np.array([0, 0, 0, 1]),
                num_labels=2
                ),
            np.array([
                [2, 1],
                [0, 1]
            ])
            )
        
        np.testing.assert_equal(
            evaluations.get_confusion_matrix(
                predicted_labels=np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]),
                true_labels     =np.array([1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]),
                num_labels=2
                ),
            np.array([
                [5, 2],
                [1, 4]
            ])
            )
        
        np.testing.assert_equal(
            evaluations.get_confusion_matrix(
                predicted_labels=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                true_labels     =np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                num_labels=3
                ),
            np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])
            )
        
        np.testing.assert_equal(
            evaluations.get_confusion_matrix(
                predicted_labels=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                true_labels     =np.array([0, 0, 0, 0, 1, 2, 0, 1, 2]),
                num_labels=3),
            np.array([
                [3, 1, 1],
                [0, 1, 1],
                [0, 1, 1]
            ])
            )
    
    #########################################
    def test_get_confusion_map(self):
        np.testing.assert_equal(
            evaluations.get_confusion_map(
                predicted_labels=np.array([0, 0, 1, 1]),
                true_labels     =np.array([0, 1, 0, 1]),
                label_index=0
                ),
            np.array([0, 0, 1, 0])
            )
            
        np.testing.assert_equal(
            evaluations.get_confusion_map(
                predicted_labels=np.array([0, 0, 1, 1]),
                true_labels     =np.array([0, 1, 0, 1]),
                label_index=1
                ),
            np.array([1, 0, 1, 1])
            )
            
        np.testing.assert_equal(
            evaluations.get_confusion_map(
                predicted_labels=np.array([[0, 0], [1, 1]]),
                true_labels     =np.array([[0, 1], [0, 1]]),
                label_index=0
                ),
            np.array([[0, 0], [1, 0]])
            )
            
        np.testing.assert_equal(
            evaluations.get_confusion_map(
                predicted_labels=np.array([[0, 0], [1, 1]]),
                true_labels     =np.array([[0, 1], [0, 1]]),
                label_index=1
                ),
            np.array([[1, 0], [1, 1]])
            )
    
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
            ev.get_global_results(),
            ([4/6, 2/4], 6/10)
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
            ev.get_global_results(),
            ([4/8, 2/6], 6/10)
            )
    
    
if __name__ == '__main__':
    unittest.main()