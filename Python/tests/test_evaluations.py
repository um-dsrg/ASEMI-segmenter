import unittest
import numpy as np
import os
import sys
sys.path.append(os.path.join('..', 'lib'))
import evaluations

#########################################
class Evaluations(unittest.TestCase):
    
    #########################################
    def test_get_classification_accuracies(self):
        np.testing.assert_equal(
                evaluations.get_classification_accuracies(
                        predicted_labels=np.array([ 0, 0, 1, 1 ]),
                        true_labels     =np.array([ 0, 1, 0, 1 ]),
                        num_labels=2
                    ),
                [ 1/2, 1/2 ]
            )
        
        np.testing.assert_equal(
                evaluations.get_classification_accuracies(
                        predicted_labels=np.array([ 0, 0, 0, 0, 1, 1 ]),
                        true_labels     =np.array([ 0, 0, 0, 1, 0, 1 ]),
                        num_labels=2
                    ),
                [ 3/4, 1/2 ]
            )
    
    #########################################
    def test_get_intersection_over_union(self):
        np.testing.assert_equal(
                evaluations.get_intersection_over_union(
                        predicted_labels=np.array([ [ 0, 1 ], [ 0, 1 ] ]),
                        true_labels     =np.array([ [ 0, 1 ], [ 1, 0 ] ]),
                        num_labels=2
                    ),
                [ 1/3, 1/3 ]
            )
        
        np.testing.assert_equal(
                evaluations.get_intersection_over_union(
                        predicted_labels=np.array([ [ 1, 1 ], [ 0, 1 ] ]),
                        true_labels     =np.array([ [ 0, 1 ], [ 1, 0 ] ]),
                        num_labels=2
                    ),
                [ 0, 1/4 ]
            )
    
    
if __name__ == '__main__':
    unittest.main()