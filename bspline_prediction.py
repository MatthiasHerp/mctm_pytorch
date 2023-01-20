import torch
from bspline.BSpline import BSpline as Bspline
import numpy as np

# Equivalent to bernstein_prediction for bsplines
# Based on the code in bspline folder originally from: https://github.com/JoKerDii/bspline-PyTorch-blocks/tree/6f62c2e59ee6566d8a6cd9272da1922409f896be
#def bspline_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0):
#
#    params_restricted = params_a.clone()
#    n = degree + 1
#
#    bspline = Bspline(start=polynomial_range[0], end=polynomial_range[1], n_bases=n, spline_order=3)
#    bspline_basis_matrix = bspline.predict(input_a)
#
#    return torch.matmul(bspline_basis_matrix,params_restricted)

def B(x, k, i, t):
    if k == 0:
       return torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
    if t[i+k] == t[i]:
       c1 = torch.FloatTensor([0.0])
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = torch.FloatTensor([0.0])
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

def Naive(x, t, c, p):
    n = len(t) - p - 1 -1
    assert (n >= p+1) and (len(c) >= n)
    pred = x.clone()
    for obs_num in range(x.size(0)):
        pred[obs_num] = sum(c[i] * B(x[obs_num], p, i, t) for i in range(n))

    return pred

def deBoor(x, t, c, p):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    n = t.size(0) - 2*p

    pred = x.clone()
    for i in range(x.size(0)):

        k = torch.searchsorted(t, x[i]) - 1
        k[k > (n - 1)] = 2 + 1
        k[k > (n - 1)] = 2 + (n - 1) - 1

        d = [c[j + k - p] for j in range(0, p + 1)]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (x[i] - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        pred[i] = d[p]

    return pred

class deBoor():
    def __init__(self,t,c,p,x):
        self.t=t
        self.c=c
        self.p=p
        self.n = t.size(0) - 2 * self.p

    def compute_k(self, x):

        k = torch.searchsorted(self.t, x) - 1
        k[k > (self.n - 1)] = 2 + 1
        k[k > (self.n - 1)] = 2 + (self.n - 1) - 1

        return k


    def compute_prediction(self, x, k):
        """Evaluates S(x).

        Arguments
        ---------
        k: Index of knot interval that contains x.
        x: Position.
        t: Array of knot positions, needs to be padded as described above.
        c: Array of control points.
        p: Degree of B-spline.
        """

        d = [self.c[j + k - self.p] for j in range(0, self.p + 1)]

        for r in range(1, self.p + 1):
            for j in range(self.p, r - 1, -1):
                alpha = (x - self.t[j + k - self.p]) / (self.t[j + 1 + k - r] - self.t[j + k - self.p])
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[self.p]

import functorch

def run_deBoor(x, t, c, p):
    deBoor_obj = deBoor(t=t, c=c, p=p, x=x)
    deBorr_func_vectorized = functorch.vmap(deBoor_obj.compute_prediction)

    #k = torch.searchsorted(t, x) - 1
    #n = t.size(0) - 2 * 2
    #k[k > (n - 1)] = 2 + 1
    #k[k > (n - 1)] = 2 + (n - 1) - 1
    k = deBoor_obj.compute_k(x)
#
    return deBorr_func_vectorized(torch.unsqueeze(x,0), torch.unsqueeze(k,0)).squeeze()

from bspline.spline_utils import torch_cr_spl_predict as torch_cr_spl_predict
def cubic_bspline(x, knots, F, S):
    #knots = torch.linspace(min, max, n_bases)

    x = torch_cr_spl_predict(x, knots, F)

    return x




# Bspline Prediction using the deBoor algorithm
def bspline_prediction(params_a, input_a, degree, polynomial_range, F, S, monotonically_increasing=False, derivativ=0,calc_method="deBoor"):

    params_restricted = params_a.clone().contiguous()
    input_a_clone = input_a.clone().contiguous()
    n = degree+1
    distance_between_knots = (polynomial_range[1] - polynomial_range[0]) / (n-1)

    knots = torch.tensor(np.linspace(polynomial_range[0]-2*distance_between_knots,
                                     polynomial_range[1]+2*distance_between_knots,
                                     n+4), dtype=torch.float32)

    #intervall_correspondance = torch.searchsorted(knots, input_a)
    #intervall_correspondance[intervall_correspondance > (n-1)] = 2 + 1
    #intervall_correspondance[intervall_correspondance > (n-1)] = 2 + (n -1) -1
    #prediction = torch.tensor([deBoor(k=k - 1,
    #                                  x=x,
    #                                  t=knots,
    #                                  c=params_restricted,
    #                                  p=2)
    #                           for (x, k) in zip(input_a, intervall_correspondance)])
    if calc_method == 'deBoor':
        prediction = run_deBoor(x=input_a_clone,
                            t=knots,
                            c=params_restricted,
                            p=2)
    elif calc_method == 'Naive':
        prediction = Naive(x=input_a_clone,
                           t=knots,
                           c=params_restricted,
                           p=2)
    elif calc_method == 'cubic_bspline':
        prediction = cubic_bspline(x=input_a_clone,
                                   knots=knots,
                                   F=F,
                                   S=S)
    elif calc_method == 'torch_bspline':
        bspline = Bspline(start=polynomial_range[0], end=polynomial_range[1], n_bases=n, spline_order=3)
        bspline_basis_matrix = bspline.predict(input_a)
        prediction = torch.matmul(bspline_basis_matrix, params_restricted)

    return prediction

#Code to test how fast different implementations are
#def B(x, k, i, t):
#    if k == 0:
#       return 1.0 if t[i] <= x < t[i+1] else 0.0
#    if t[i+k] == t[i]:
#       c1 = 0.0
#    else:
#       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#    if t[i+k+1] == t[i+1]:
#       c2 = 0.0
#    else:
#       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#    return c1 + c2
#
#def bspline(x, t, c, k):
#    n = len(t) - k - 1
#    assert (n >= k+1) and (len(c) >= n)
#    return sum(c[i] * B(x, k, i, t) for i in range(n))
#data = np.array(range(10000))
#
#import time
#from scipy.interpolate import BSpline
#k = 2
#t = [0, 1, 2, 3, 4, 5, 6]
#c = [-1, 2, 0, -1]
#
##scipy
#start = time.time()
#spl = BSpline(t, c, k)
#spl(data)
#end = time.time()
#end-start
#
##vanilla
#start = time.process_time()
#[bspline(x, t, c, k) for x in data]
#end = time.process_time()
#end-start
#
##pytorch absed on scipy
#start = time.process_time()
#bspline = Bspline(start=0, end=10000, n_bases=4, spline_order=2)
#bspline.predict(torch.tensor(data))
#end = time.process_time()
#end-start