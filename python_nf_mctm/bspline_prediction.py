import torch
import numpy as np
import functorch
from python_nf_mctm.splines_utils import adjust_ploynomial_range, ReLULeR, custom_sigmoid

def x_in_intervall(x, i, t):
    # if t[i] <= x < t[i+1] then this is one, otherwise zero
    return torch.where(t[i] <= x,
                       torch.FloatTensor([1.0], device=x.device),
                       torch.FloatTensor([0.0], device=x.device)) * \
           torch.where(x < t[i+1],
                       torch.FloatTensor([1.0], device=x.device),
                       torch.FloatTensor([0.0], device=x.device))

def B(x, k, i, t):
    """

    :param x: observatioon vector
    :param k: degree of the basis function
    :param i:
    :param t: knots vector
    :return:
    """
    # added due to derivativ computation of Bspline
    if k < 0:
        return torch.FloatTensor([0.0])
    if k == 0:
       return x_in_intervall(x, i, t) #torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
    if t[i+k] == t[i]:
       c1 = torch.FloatTensor([0.0])
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = torch.FloatTensor([0.0])
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

def B_vmap(x, k, i, t, knot):
    if knot < t[i-k] or knot > t[i+k]:
       return torch.FloatTensor([0.0])
    if k == 0:
       return x_in_intervall(x, i, t) #torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
    if t[i+k] == t[i]:
       c1 = torch.FloatTensor([0.0])
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = torch.FloatTensor([0.0])
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2

#def k_is_zero(k):
#    # if k == 0 then this is zero, otherwise one
#    return torch.where(k==torch.tensor(0), torch.FloatTensor([0.0]), torch.FloatTensor([1.0]))
#
#def x_in_intervall(x, i, t):
#    # if t[i] <= x < t[i+1] then this is one, otherwise zero
#    return torch.where(t[i] <= x,
#                       torch.FloatTensor([1.0]),
#                       torch.FloatTensor([0.0])) * \
#           torch.where(x < t[i+1],
#                       torch.FloatTensor([1.0]),
#                       torch.FloatTensor([0.0]))
#
#def c1(x, k, i, t):
#    # computes first part of the bspline
#    return torch.where(k>torch.tensor(0),
#                       torch.where(t[i + k] == t[i],
#                                   torch.FloatTensor([0.0]),
#                                    (x - t[i]) / (t[i + k] - t[i]) * B_vmap(x, k - 1, i, t)),
#                          torch.FloatTensor([0.0]))
#
#def c2(x, k, i, t):
#    # computes second part of the bspline
#    return torch.where(k>torch.tensor(0),
#                       torch.where(t[i + k + 1] == t[i + 1],
#                                   torch.FloatTensor([0.0]),
#                                   (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * B_vmap(x, k - 1, i + 1, t)),
#                          torch.FloatTensor([0.0]))
#def B_vmap(x, k, i, t):
#    print(k)
#    # computes the bspline with 2 options:
#    # 1. if k == 0 then return 1 if x is in the intervall [t[i], t[i+1]) and 0 otherwise
#    # 2. if k != 0 then return the bspline
#    return torch.where(k>torch.tensor(0),
#                c1(x, k, i, t) + c2(x, k, i, t),
#                x_in_intervall(x, i, t))
#    #return (c1(x, k, i, t) + c2(x, k, i, t)) * k_is_zero(k) + (1-k_is_zero(k)) * x_in_intervall(x, i, t)

def Naive(x, t, c, p):
    n = len(t) - p - 1 -1
    #assert (n >= p+1) and (len(c) >= n)
    #pred = x.clone()
    #for obs_num in range(x.size(0)):
    #    pred[obs_num] = sum(c[i] * B(x[obs_num], p, i, t) for i in range(n))

    pred = sum(c[i] * B(x, p, i, t) for i in range(n))

    return pred

from python_nf_mctm.bernstein_prediction import kron

def Naive_Basis(x, polynomial_range, degree, span_factor,derivativ=0, device=None):
    order = 2
    p = order
    n = degree + 1
    distance_between_knots = (polynomial_range[1] - polynomial_range[0]) * span_factor / (n - 1)

    knots = torch.linspace(polynomial_range[0] - order * distance_between_knots,
                                     polynomial_range[1] + order * distance_between_knots,
                                     n + 4, dtype=torch.float32, device=device)
    t = knots


    n = len(t) - p - 1 - 1
    if derivativ==0:
        return torch.vstack([B(x, p, i, t) for i in range(n)]).T
    elif derivativ==1:
        return torch.vstack([p*(B(x, p-1, i, t)/(t[i+p]-t[i]) - B(x, p-1, i+1, t)/(t[i+p+1]-t[i+1])) for i in range(n)]).T



def compute_multivariate_bspline_basis(input, degree, polynomial_range, span_factor, covariate=False, derivativ=0, device=None):
    # We essentially do a tensor prodcut of two splines! : https://en.wikipedia.org/wiki/Bernstein_polynomial#Generalizations_to_higher_dimension

    if covariate is not False:
        multivariate_bspline_basis = torch.empty(size=(input.size(0), (degree+1)*(degree+1), input.size(1)), device=device)
    else:
        multivariate_bspline_basis = torch.empty(size=(input.size(0), (degree+1), input.size(1)), device=device)

    for var_num in range(input.size(1)):
        input_basis = Naive_Basis(x=input[:, var_num], degree=degree, polynomial_range=polynomial_range[:, var_num], span_factor=span_factor, derivativ=derivativ, device=device)
        if covariate is not False:
            #covariate are transformed between 0 and 1 before inputting into the model
            # dont take the derivativ w.r.t to the covariate when computing jacobian of the transformation
            covariate_basis = Naive_Basis(x=covariate, degree=degree, polynomial_range=torch.tensor([0,1],device=device), span_factor=span_factor, derivativ=derivativ, device=device)
            basis = kron(input_basis, covariate_basis)
        else:
            basis = input_basis

        multivariate_bspline_basis[:,:,var_num] = basis

    return multivariate_bspline_basis

# cannot use vmap here as B(x, k, i, t) contains if statments which vmap does not support yet:
# https://github.com/pytorch/functorch/issues/257
class Naive_vmap():
    def __init__(self,t,c,p):
        self.t=t
        self.c=c
        self.p=p
        self.n = t.size(0) - 2 * self.p

    def compute_knot(self, x):

        k = torch.searchsorted(self.t, x) - 1
        k[k > (self.n - 1)] = 2 + 1
        k[k > (self.n - 1)] = 2 + (self.n - 1) - 1

        self.knots = k

    def compute_prediction(self, x):
        n = len(self.t) - self.p - 1 -1
        #assert (n >= self.p+1) and (len(self.c) >= n)
        #pred = x.clone()
        #for obs_num in range(x.size(0)):
            #knot_num = knot[torch.tensor(obs_num)[None]][0]
            #pred[obs_num] = sum(self.c[i] * B(x[obs_num], self.p, i, self.t) for i in range(n)) #knot_num-self.p,knot_num+self.p
        #Works
        #pred = sum(self.c[i] * torch.vstack([B_vmap(x[obs_num], self.p, 0, self.t, self.knots[obs_num]) for obs_num in range(x.size(0))]).squeeze() for i in range(n))
        # Does not work
        #pred = sum(self.c[i] * functorch.vmap(B_vmap)(x, self.p, i, self.t, self.knots) for i in range(n))

        def B_vmap(i,obs_num):
            if self.knots[obs_num] < self.t[i - self.p] or self.knots[obs_num] > self.t[i + self.p]:
                return torch.FloatTensor([0.0])
            if k == 0: #TODO: error here
                return x_in_intervall(x[obs_num], i,
                                      self.t)  # torch.FloatTensor([1.0]) if t[i] <= x < t[i+1] else torch.FloatTensor([0.0])
            if self.t[i + self.p] == self.t[i]:
                c1 = torch.FloatTensor([0.0])
            else:
                c1 = (x[obs_num] - self.t[i]) / (self.t[i + self.p] - self.t[i]) * B(x[obs_num], self.k - 1, i, self.t)
            if self.t[i + self.p + 1] == self.t[i + 1]:
                c2 = torch.FloatTensor([0.0])
            else:
                c2 = (self.t[i + self.p + 1] - x[obs_num]) / (self.t[i + self.p + 1] - self.t[i + 1]) * B(x[obs_num], self.p - 1, i + 1, self.t)
            return c1 + c2

        pred = sum(self.c[i] * torch.vstack([B_vmap(i,obs_num) for obs_num in range(x.size(0))]).squeeze() for i in range(n))

        return pred

def run_Naive_vmap(x, t, c, p):
    Naive_vmap_obj = Naive_vmap(t=t, c=c, p=p)

    Naive_vmap_obj.compute_knot(x)

    Naive_vmap_func_vectorized = functorch.vmap(Naive_vmap_obj.compute_prediction)

    return Naive_vmap_func_vectorized(torch.unsqueeze(x, 0)).squeeze()

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

# Bspline Prediction using the deBoor algorithm
def bspline_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0, return_penalties=False, calc_method='deBoor', span_factor=0.1, span_restriction=None,
                       covariate=False, params_covariate=False, device=None):

    # Adjust polynomial range to be a bit wider
    # Empirically found that this helps with the fit
    polynomial_range = adjust_ploynomial_range(polynomial_range, span_factor)

    order=2
    params_restricted = params_a.clone().contiguous()
    input_a_clone = input_a.clone().contiguous()
    n = degree+1
    distance_between_knots = (polynomial_range[1] - polynomial_range[0]) / (n-1)

    knots = torch.linspace(polynomial_range[0]-order*distance_between_knots,
                                     polynomial_range[1]+order*distance_between_knots,
                                     n+4, dtype=torch.float32, device=device)

    #input_a_clone = (torch.sigmoid(input_a_clone/((polynomial_range[1] - polynomial_range[0])) * 10) - 0.5) * (polynomial_range[1] - polynomial_range[0])/2
    #input_a_clone = custom_sigmoid(input=input_a_clone, min=polynomial_range[0], max=polynomial_range[1])

    if span_restriction == "sigmoid":
        input_a_clone = custom_sigmoid(input_a_clone, polynomial_range)
    elif span_restriction == "reluler":
        reluler = ReLULeR(polynomial_range)
        input_a_clone = reluler.forward(input_a_clone)


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

    elif calc_method == "Naive_vmap":
        prediction = run_Naive_vmap(x=input_a_clone,
                                    t=knots,
                                    c=params_restricted,
                                    p=order)

    # Adding Covariate in a GAM manner
    if covariate is not False:
        params_covariate_restricted = params_covariate.clone().contiguous()

        #if dev is not False:
        #    params_covariate_restricted.to(dev)

        knots_covariate = torch.linspace(0 - order * 1,
                                         1 + order * 1,
                                         n + 4, dtype=torch.float32, device=device)

        prediction_covariate = run_deBoor(x=covariate,
                                t=knots_covariate,
                                c=params_covariate_restricted,
                                p=order)

        prediction = prediction * prediction_covariate


    if prediction.isnan().sum() > 0:
       print("prediction contains NaNs")
       print("prediction is nan:",prediction[prediction.isnan()])
       print("knots:",knots)


    if return_penalties:
        second_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=2)**2)
        first_order_ridge_pen = torch.sum(torch.diff(params_restricted,n=1)**2)
        param_ridge_pen = torch.sum(params_restricted**2)

        # Adding Covariate parameter penalisation values
        if covariate is not False:
            second_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=2) ** 2)
            first_order_ridge_pen += torch.sum(torch.diff(params_covariate_restricted, n=1) ** 2)
            param_ridge_pen += torch.sum(params_covariate_restricted ** 2)


        return prediction, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    else:
        return prediction