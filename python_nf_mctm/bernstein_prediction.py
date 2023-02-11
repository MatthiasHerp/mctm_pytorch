import torch
from python_nf_mctm.splines_utils import adjust_ploynomial_range

# https://github.com/pytorch/pytorch/issues/47841
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.binom.html
def torch_binom(n, v):
    #mask = n.detach() >= v.detach()
    #n = mask * n
    #v = mask * v
    a = torch.lgamma(n + 1) - torch.lgamma((n - v) + 1) - torch.lgamma(v + 1)
    return torch.exp(a)# * mask

# https://en.wikipedia.org/wiki/Bernstein_polynomial
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BPoly.html
def b(v, n, x):
    return torch_binom(n, v) * x**v * (1 - x)**(n - v)

def bernstein_prediction(params_a, input_a, degree, polynomial_range, monotonically_increasing=False, derivativ=0, span_factor=0.1,
                         covariate=None,params_covariate=None):

    # Adjust polynomial range to be a bit wider
    # Empirically found that this helps with the fit
    polynomial_range = adjust_ploynomial_range(polynomial_range, span_factor)

    # Restricts Monotonically increasing by insuring that params increase
    if monotonically_increasing:

        params_restricted = params_a.clone()
        #relu_obj = nn.ReLU()
        params_restricted[1:] = torch.matmul(torch.exp(params_a[1:]),torch.triu(torch.ones(degree,degree))) + params_a[0] #abs or exp
        #.double() otherwise I started getting the folowing error:
        #{RuntimeError}Expected object of scalar type Double but got scalar type Float for argument #2 'mat2' in call to _th_mm
        #alternativ set dtype=torch.float64 in torch.ones(degree,degree)

        # cannot use exp as it destabilizes the optimisation using LBFGS, get inf for params fast?
        # however with exp works really well and fast with adam
        # Relu works for both, results are good except at lower end there is a cutoff somehow

        # We allso need to restrict covariates to ensure the sum of the gam is monotonically increasing
        if covariate is not None:
            params_covariate_restricted = params_covariate.clone()

            params_covariate_restricted[1:] = torch.matmul(torch.exp(params_covariate[1:]), torch.triu(torch.ones(degree, degree))) + \
                                    params_covariate[0]

    else:
        params_restricted = params_a.clone()
        params_covariate_restricted = params_covariate.clone()
    n = degree
    #return sum((params[1] + sum(torch.abs(params[1:(v-1)]))) * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input) for v in range(n+1))

    # penalities
    second_order_ridge_pen = 0
    first_order_ridge_pen = 0
    param_ridge_pen = 0


    normalizing_range = polynomial_range[1] - polynomial_range[0]
    input_a = (input_a - polynomial_range[0]) / (normalizing_range)

    if derivativ == 0:
        output = sum(params_restricted[v] * b(torch.FloatTensor([v]), torch.FloatTensor([n]), input_a) for v in range(n+1)) #before we had: params_restricted[v-1]

    elif derivativ == 1:
        #output = sum(params_restricted[v] * b(torch.FloatTensor([v - 1]), torch.FloatTensor([n]), input_a) * (
        #            torch.FloatTensor([v]) - torch.FloatTensor([n]) * input_a) for v in range(n + 1))
        # My derivativ see goodnotes
        output = 1/normalizing_range * sum(params_restricted[v] * torch.FloatTensor([n]) * (b(torch.FloatTensor([v-1]), torch.FloatTensor([n-1]), input_a) -
                                                                                            b(torch.FloatTensor([v]), torch.FloatTensor([n-1]), input_a)) for v in range(n+1))
        # The Bernstein polynomial basis: A centennial retrospective p.391 (17)

    # In a GAM we add the covariate effect as a spline model
    if covariate is not None:
        covariate_effect = sum(params_covariate_restricted[v] * b(torch.FloatTensor([v]), torch.FloatTensor([n]), covariate) for v in
                     range(n + 1))
        output = output + covariate_effect

    return output, second_order_ridge_pen, first_order_ridge_pen, param_ridge_pen
    #return (output + polynomial_range[0]) / (polynomial_range[1] - polynomial_range[0])
