import torch
import time
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

        Naive_vmap_prediction = bspline_prediction(params,
                                                  data,
                                                  degree,
                                                  polynomial_range,
                                                  calc_method='Naive_vmap')

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

        Naive_vmap_prediction = bspline_prediction(params,
                                                   data,
                                                   degree,
                                                   polynomial_range,
                                                   calc_method='Naive_vmap')

        self.assertTrue(torch.allclose(Naive_vmap_prediction, Naive_prediction, 1e-5))

    def test_3(self):
        degree = 10
        params = torch.arange(0,degree + 1)
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

        Naive_vmap_prediction = bspline_prediction(params,
                                                   data,
                                                   degree,
                                                   polynomial_range,
                                                   calc_method='Naive_vmap')

        self.assertTrue(torch.allclose(Naive_vmap_prediction, Naive_prediction, 1e-5))

    def test_4(self):
        degree = 40
        params = torch.arange(0, degree + 1)
        polynomial_range = torch.tensor([-7, 7])
        data = torch.linspace(-5, 5, 100000)

        start = time.time()
        deBoor_prediction = bspline_prediction(params,
                                               data,
                                               degree,
                                               polynomial_range,
                                               calc_method='deBoor')
        end = time.time()
        print("Deboor run time:",end - start)

        start = time.time()
        Naive_prediction = bspline_prediction(params,
                                              data,
                                              degree,
                                              polynomial_range,
                                              calc_method='Naive')
        end = time.time()
        print("Naive run time:",end - start)

        start = time.time()
        Naive_vmap_prediction = bspline_prediction(params,
                                                   data,
                                                   degree,
                                                   polynomial_range,
                                                   calc_method='Naive_vmap')
        end = time.time()
        print("Naive vmap run time:",end - start)

        self.assertTrue(torch.allclose(deBoor_prediction, Naive_prediction, 1e-5))


if __name__ == '__main__':
    unittest.main()


#TODO:
# 1. write test script to check speed of different methods
# 2. add setting bspline value to zero for far way knots when computing Naive_vmap