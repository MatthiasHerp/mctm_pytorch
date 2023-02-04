import torch
import numpy as np
import functorch
from reluler_layer import ReLULeR

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

# cannot use vmap here as B(x, k, i, t) contains if statments which vmap does not support yet:
# https://github.com/pytorch/functorch/issues/257
#class Naive():
#    def __init__(self,t,c,p):
#        self.t=t
#        self.c=c
#        self.p=p
#        self.n = t.size(0) - 2 * self.p
#
#    def compute_prediction(self, x):
#        n = len(self.t) - self.p - 1 -1
#        assert (n >= self.p+1) and (len(self.c) >= n)
#        pred = x.clone()
#        for obs_num in range(x.size(0)):
#            pred[obs_num] = sum(self.c[i] * B(x[obs_num], self.p, i, self.t) for i in range(n))
#
#        return pred

#def run_Naive(x, t, c, p):
#    Naive_obj = Naive(t=t, c=c, p=p)
#    Naive_func_vectorized = functorch.vmap(Naive_obj.compute_prediction)
#
#    return Naive_func_vectorized(torch.unsqueeze(x,0)).squeeze()

class deBoor():
    def __init__(self,t,c,p):
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

# Runs DeBoor Algorithm vectorized using functorch
# (much faster (30 sec vs 8:30min) alternative to using a for loop or list comprehension)
def run_deBoor(x, t, c, p):
    deBoor_obj = deBoor(t=t, c=c, p=p)
    deBorr_func_vectorized = functorch.vmap(deBoor_obj.compute_prediction)
    k = deBoor_obj.compute_k(x)

    return deBorr_func_vectorized(torch.unsqueeze(x,0), torch.unsqueeze(k,0)).squeeze()

def custom_sigmoid(input: torch.Tensor, min: float, max: float):
    input_01 = (input - min) / (max - min)
    input_11 = input_01 * 2 - 1
    input_bounded_01 = 1 / (1 + torch.exp(-input_11 * 4 ))
    input_bounded = input_bounded_01 * (max - min) + min

    return input_bounded


# Bspline Prediction using the deBoor algorithm
def bspline_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0, return_penalties=False, calc_method='deBoor'):

    order=2
    params_restricted = params_a.clone().contiguous()
    input_a_clone = input_a.clone().contiguous()
    n = degree+1
    distance_between_knots = (polynomial_range[1] - polynomial_range[0]) / (n-1)

    knots = torch.tensor(np.linspace(polynomial_range[0]-order*distance_between_knots,
                                     polynomial_range[1]+order*distance_between_knots,
                                     n+4), dtype=torch.float32)
    #if 0.0 in knots:
    #    print("0 in knots") #0.25, 0.5, 0.1, 0.3, 0
    #else:
    #    print("0 not in knots") #0.2

    #ReLULeR_obj = ReLULeR(polynomial_range_abs=polynomial_range[1])
    #input_a_clone = ReLULeR_obj.forward(input_a_clone)
    #input_a_clone = (torch.sigmoid(input_a_clone/((polynomial_range[1] - polynomial_range[0])) * 10) - 0.5) * (polynomial_range[1] - polynomial_range[0])/2
    #input_a_clone = custom_sigmoid(input=input_a_clone, min=polynomial_range[0], max=polynomial_range[1])

    if calc_method == "deBoor":
        prediction = run_deBoor(x=input_a_clone,
                                t=knots,
                                c=params_restricted,
                                p=order)
    elif calc_method == "Naive":
        prediction = Naive(x=input_a_clone,
                           t=knots,
                           c=params_restricted,
                           p=order)


    if prediction.isnan().sum() > 0:
       print("prediction contains NaNs")
       print("prediction is nan:",prediction[prediction.isnan()])
       print("knots:",knots)


    if return_penalties:
        second_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=2)**2)
        first_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=1)**2)
        param_ridge_pen = torch.sum(params_restricted**2)
        return prediction, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    else:
        return prediction