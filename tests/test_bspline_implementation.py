import torch
import numpy as np
import unittest
from python_nf_mctm.bspline_prediction import bspline_prediction

class TestPrediction(unittest.TestCase):
    """
    test_equal_across_methods:
    Test that deBoor and Naive Bspline implementation approximately give the same prediction.
    We use values not close to the border of the spline to avoid numerical issues.
    At the border of the spline the implementations start to differ in 0.01 maginitudes.
    """
    def test_1(self):
        print("starting test")
        degree = 10
        params = torch.tensor(np.repeat(0.1,degree+1), dtype=torch.float32)
        polynomial_range = torch.tensor([-7,7])
        data = torch.linspace(-5,5,100)

        deBoor_prediction = bspline_prediction(params,
                                               data,
                                               degree,
                                               polynomial_range,
                                               calc_method='deBoor')

        Naive_prediction = bspline_prediction(params,
                                               data,
                                               degree,
                                               polynomial_range,
                                               calc_method='Naive')

        self.assertTrue(torch.allclose(deBoor_prediction, Naive_prediction,1e-5))

    def test_2(self):
        degree = 10
        params = torch.linspace(0,1, degree + 1)
        polynomial_range = torch.tensor([-7, 7])
        data = torch.linspace(-5, 5, 100)

        deBoor_prediction = bspline_prediction(params,
                                               data,
                                               degree,
                                               polynomial_range,
                                               calc_method='deBoor')

        Naive_prediction = bspline_prediction(params,
                                              data,
                                              degree,
                                              polynomial_range,
                                              calc_method='Naive')

if __name__ == '__main__':
    unittest.main()