import torch
from bspline.BSpline import BSpline as Bspline

# Equivalent to bernstein_prediction for bsplines
# Based on the code in bspline folder originally from: https://github.com/JoKerDii/bspline-PyTorch-blocks/tree/6f62c2e59ee6566d8a6cd9272da1922409f896be
def bspline_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0):

    params_restricted = params_a.clone()
    n = degree

    bspline = Bspline(start=polynomial_range[0], end=polynomial_range[1], n_bases=n, spline_order=3)
    bspline_basis_matrix = bspline.predict(input_a)

    return torch.matmul(bspline_basis_matrix,params_restricted)

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