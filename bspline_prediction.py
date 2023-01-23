import torch
import numpy as np
import functorch

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

# Runs DeBoor Algorithm vectorized using functorch
# (much faster (30 sec vs 8:30min) alternative to using a for loop or list comprehension)
def run_deBoor(x, t, c, p):
    deBoor_obj = deBoor(t=t, c=c, p=p, x=x)
    deBorr_func_vectorized = functorch.vmap(deBoor_obj.compute_prediction)
    k = deBoor_obj.compute_k(x)

    return deBorr_func_vectorized(torch.unsqueeze(x,0), torch.unsqueeze(k,0)).squeeze()

# Bspline Prediction using the deBoor algorithm
def bspline_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0, return_penalties=False):

    params_restricted = params_a.clone().contiguous()
    input_a_clone = input_a.clone().contiguous()
    n = degree+1
    distance_between_knots = (polynomial_range[1] - polynomial_range[0]) / (n-1)

    knots = torch.tensor(np.linspace(polynomial_range[0]-2*distance_between_knots,
                                     polynomial_range[1]+2*distance_between_knots,
                                     n+4), dtype=torch.float32)

    prediction = run_deBoor(x=input_a_clone,
                            t=knots,
                            c=params_restricted,
                            p=2)

    if return_penalties:
        second_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=2)**2)
        first_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=1)**2)
        param_ridge_pen = torch.sum(params_restricted**2)
        return prediction, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    else:
        return prediction